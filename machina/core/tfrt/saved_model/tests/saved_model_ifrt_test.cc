/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, July 12, 2025.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201,
 * Middletown, DE 19709, New Castle County, USA.
 *
 */

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "machina/compiler/mlir/tfrt/transforms/ifrt/ifrt_backend_compiler.h"
#include "machina/compiler/mlir/tfrt/translate/tfrt_compile_options.h"
#include "machina/xla/python/ifrt/client.h"
#include "machina/xla/python/ifrt/test_util.h"
#include "machina/xla/tsl/framework/test_util/mock_serving_device_selector.h"
#include "machina/xla/tsl/lib/core/status_test_util.h"
#include "machina/core/framework/tensor.h"
#include "machina/core/platform/resource_loader.h"
#include "machina/core/tfrt/ifrt/checkpoint_loader.h"
#include "machina/core/tfrt/ifrt/ifrt_model_context.h"
#include "machina/core/tfrt/ifrt/ifrt_model_restore_context.h"
#include "machina/core/tfrt/ifrt/ifrt_serving_core_selector.h"
#include "machina/core/tfrt/runtime/runtime.h"
#include "machina/core/tfrt/saved_model/saved_model.h"
#include "machina/core/tfrt/saved_model/saved_model_testutil.h"
#include "tsl/platform/env.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/threadpool.h"
#include "tfrt/host_context/concurrent_work_queue.h"  // from @tf_runtime

namespace machina {
namespace tfrt_stub {
namespace {

tsl::thread::ThreadPool& GetThreadPool() {
  constexpr int kMaxParallelism = 16;
  static tsl::thread::ThreadPool* thread_pool =
      new tsl::thread::ThreadPool(tsl::Env::Default(), tsl::ThreadOptions(),
                                  "IfrtSharding", kMaxParallelism);
  return *thread_pool;
}

TEST(SavedModelIfrt, Basic) {
  std::string saved_model_dir = machina::GetDataDependencyFilepath(
      "machina/core/tfrt/saved_model/tests/toy_v2");

  auto runtime =
      machina::tfrt_stub::Runtime::Create(/*num_inter_op_threads=*/4);

  // Create contexts required for the compiler execution.
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::ifrt::Client> client,
                          xla::ifrt::test_util::GetClient());

  auto work_queue = tfrt::CreateMultiThreadedWorkQueue(
      /*num_threads=*/4, /*num_blocking_threads=*/4);

  tsl::test_util::MockServingDeviceSelector selector;
  ifrt_serving::IfrtServingCoreSelector core_selector(
      &selector, client->addressable_device_count());

  // Use IFRT compiler
  runtime->AddCreateRuntimeResourceFn(
      [&](machina::tfrt_stub::ModelRuntimeContext& model_context) {
        model_context.resource_context()
            .CreateResource<machina::ifrt_serving::IfrtModelContext>(
                "IfrtModelContext", client, &core_selector, &GetThreadPool(),
                /*compilation_environment_proto=*/nullptr);

        machina::ifrt_serving::IfrtModelContext* ifrt_model_context =
            (*model_context.resource_context()
                  .GetResource<machina::ifrt_serving::IfrtModelContext>(
                      "IfrtModelContext"));
        ifrt_model_context->set_checkpoint_loader_queue(work_queue.get());
        model_context.resource_context()
            .CreateResource<machina::ifrt_serving::IfrtModelRestoreContext>(
                ifrt_serving::kIfrtModelRestoreContextName,
                std::make_unique<machina::ifrt_serving::CheckpointLoader>(
                    &ifrt_model_context->GetRestoreTensorRegistry(),
                    ifrt_model_context->checkpoint_loader_queue()));

        return absl::OkStatus();
      });
  machina::ifrt_serving::IfrtBackendCompiler ifrt_compiler;

  auto options = DefaultSavedModelOptions(runtime.get());
  options.graph_execution_options.enable_mlrt = true;
  options.enable_lazy_loading = true;
  options.lazy_loading_use_graph_executor = true;
  options.graph_execution_options.compile_options.backend_compiler =
      &ifrt_compiler;

  TF_ASSERT_OK_AND_ASSIGN(
      auto saved_model, SavedModelImpl::LoadSavedModel(options, saved_model_dir,
                                                       /*tags=*/{"serve"}));

  // Set input 'x' to [[1, 1, 1]]
  std::vector<machina::Tensor> inputs;
  inputs.push_back(
      CreateTfTensor<int32_t>(/*shape=*/{1, 3}, /*data=*/{1, 1, 1}));

  tfrt::SavedModel::RunOptions run_options;

  std::vector<machina::Tensor> outputs;
  TF_ASSERT_OK(
      saved_model->Run(run_options, "serving_default", inputs, &outputs));
  ASSERT_EQ(outputs.size(), 1);

  EXPECT_THAT(GetTfTensorData<int32_t>(outputs[0]),
              ::testing::ElementsAreArray({6}));
}

}  // namespace
}  // namespace tfrt_stub
}  // namespace machina
