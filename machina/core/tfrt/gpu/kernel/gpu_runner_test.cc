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
#if GOOGLE_CUDA || MACHINA_USE_ROCM
#include "machina/core/tfrt/gpu/kernel/gpu_runner.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "machina/cc/framework/scope.h"
#include "machina/cc/ops/function_ops.h"
#include "machina/cc/ops/math_ops.h"
#include "machina/xla/tsl/framework/serving_device_selector_policies.h"
#include "machina/core/common_runtime/gpu/gpu_serving_device_selector.h"
#include "machina/core/framework/device_base.h"
#include "machina/core/framework/function.h"
#include "machina/core/framework/graph_to_functiondef.h"
#include "machina/core/framework/node_def_builder.h"
#include "machina/core/framework/tensor_testutil.h"
#include "machina/core/framework/types.h"
#include "machina/core/lib/core/status_test_util.h"
#include "machina/core/platform/errors.h"
#include "machina/core/platform/status.h"
#include "machina/core/platform/statusor.h"
#include "machina/core/runtime_fallback/kernel/kernel_fallback_compat_request_state.h"
#include "machina/core/tfrt/fallback/fallback_state.h"
#include "tfrt/host_context/concurrent_work_queue.h"  // from @tf_runtime
#include "tfrt/host_context/diagnostic.h"  // from @tf_runtime
#include "tfrt/host_context/function.h"  // from @tf_runtime
#include "tfrt/host_context/host_allocator.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime

namespace machina {
namespace gpu {
namespace {
// TODO(b/297939691): Change this to greater than 1 after PJRT device supports
// both logical and physical device IDs.
constexpr int kNumVirtualGpuDevices = 1;
constexpr char kFunctionName[] = "foo";

StatusOr<std::unique_ptr<Graph>> SampleGraphAddXY() {
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto a = ops::_Arg(scope.WithOpName("A"), DT_INT32, 0);
  auto b = ops::_Arg(scope.WithOpName("B"), DT_INT32, 1);
  auto c = ops::Add(scope.WithOpName("C"), a, b);
  auto d = ops::_Retval(scope.WithOpName("D"), c, 0);
  TF_RETURN_IF_ERROR(scope.ToGraph(graph.get()));
  return graph;
}

StatusOr<FunctionDef> SampleFunctionAddXY(const std::string& name) {
  TF_ASSIGN_OR_RETURN(auto graph, SampleGraphAddXY());
  FunctionDef fdef;
  TF_RETURN_IF_ERROR(GraphToFunctionDef(*graph, name, &fdef));
  return fdef;
}

Status GetDevices(const machina::tfd::KernelFallbackCompatRequestState*
                      fallback_request_state,
                  Device** cpu_device,
                  absl::flat_hash_map<int, Device*>& gpu_devices) {
  *cpu_device = fallback_request_state->device_manager().HostCPU();
  if (!*cpu_device) {
    return absl::InternalError(
        "Fallback request state must have a valid host cpu device.");
  }
  for (Device* device :
       fallback_request_state->device_manager().ListDevices()) {
    if (device->device_type() != DEVICE_GPU) continue;
    if (!gpu_devices.try_emplace(device->parsed_name().id, device).second) {
      return absl::InternalError(absl::StrCat(
          "A device with the same device ID already exists when adding ",
          device->name()));
    }
  }
  if (gpu_devices.empty()) {
    return absl::InternalError("No GPU device is found.");
  }
  for (const auto& [id, device] : gpu_devices) {
    if (id >= gpu_devices.size()) {
      return absl::InternalError("Device IDs are not consecutive.");
    }
  }
  return OkStatus();
}

template <typename T>
Tensor CreateTensor(const TensorShape& input_shape,
                    gtl::ArraySlice<T> input_data,
                    Allocator* allocator = nullptr) {
  Tensor tensor(DataTypeToEnum<T>::value, input_shape);
  test::FillValues<T>(&tensor, input_data);
  return tensor;
}

class GpuRunnerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create FallbackState.
    machina::SessionOptions session_options;
    TF_ASSERT_OK_AND_ASSIGN(FunctionDef fdef,
                            SampleFunctionAddXY(kFunctionName));
    machina::FunctionDefLibrary fdef_lib;
    *fdef_lib.add_function() = fdef;
    TF_ASSERT_OK_AND_ASSIGN(fallback_state_, tfrt_stub::FallbackState::Create(
                                                 session_options, fdef_lib));

    // Create KernelFallbackCompatRequestState.
    std::function<void(std::function<void()>)> runner =
        [](const std::function<void()>& f) { f(); };
    tfrt_stub::OpKernelRunnerTable runner_table;
    tfd::FallbackResourceArray resource_array;
    fallback_request_state_ =
        std::make_unique<tfd::KernelFallbackCompatRequestState>(
            &runner, &fallback_state_->device_manager(), /*step_id=*/0,
            &runner_table, &resource_array,
            /*user_intra_op_threadpool=*/nullptr,
            /*model_metadata=*/std::nullopt,
            &fallback_state_->process_function_library_runtime());

    // Create execution context.
    auto host_allocator = tfrt::CreateMallocAllocator();
    auto work_queue = tfrt::CreateMultiThreadedWorkQueue(
        /*num_threads=*/2, /*num_blocking_threads=*/2);

    host_context_ = std::make_unique<tfrt::HostContext>(
        [&](const tfrt::DecodedDiagnostic& diag) {}, std::move(host_allocator),
        std::move(work_queue));
    tfrt::RequestContextBuilder req_ctx_builder =
        tfrt::RequestContextBuilder(host_context_.get(), nullptr);
    tfrt::Expected<tfrt::RCReference<tfrt::RequestContext>> req_ctx(
        std::move(req_ctx_builder).build());
    ASSERT_TRUE(!!req_ctx);
    exec_ctx_ = std::make_unique<tfrt::ExecutionContext>(std::move(*req_ctx));

    // Create a gpu runner.
    auto policy = std::make_unique<tsl::RoundRobinPolicy>();
    serving_device_selector_ = std::make_unique<GpuServingDeviceSelector>(
        kNumVirtualGpuDevices, std::move(policy));
    gpu_runner_ = std::make_unique<GpuRunner>(serving_device_selector_.get());
  }

  std::unique_ptr<tfrt_stub::FallbackState> fallback_state_;
  std::unique_ptr<tfd::KernelFallbackCompatRequestState>
      fallback_request_state_;
  std::unique_ptr<tfrt::HostContext> host_context_;
  std::unique_ptr<tfrt::ExecutionContext> exec_ctx_;
  std::unique_ptr<GpuServingDeviceSelector> serving_device_selector_;
  std::unique_ptr<GpuRunner> gpu_runner_;
};

TEST_F(GpuRunnerTest, Basic) {
  // Construct GpuRunInputs.
  GpuRunInputs run_inputs;

  toolchain::SmallVector<tfrt_stub::FallbackTensor> args;
  Tensor tensor1 = CreateTensor<int32>(TensorShape({1, 2}), {1, 2});
  Tensor tensor2 = CreateTensor<int32>(TensorShape({1, 2}), {3, 4});
  args.push_back(tfrt_stub::FallbackTensor(tensor1));
  args.push_back(tfrt_stub::FallbackTensor(tensor2));
  run_inputs.args = &args;

  run_inputs.num_outputs = 1;
  run_inputs.resource_indices = tfrt::ArrayRef<int64_t>(0);
  run_inputs.used_output_indices = tfrt::ArrayRef<int64_t>(0);
  run_inputs.func_name = kFunctionName;

  absl::flat_hash_map<int, Device*> gpu_devices;
  ASSERT_OK(GetDevices(fallback_request_state_.get(), &run_inputs.cpu_device,
                       gpu_devices));
  run_inputs.gpu_devices = &gpu_devices;
  run_inputs.fallback_request_state = fallback_request_state_.get();
  run_inputs.exec_ctx = exec_ctx_.get();

  // Run the input.
  TF_ASSERT_OK_AND_ASSIGN(
      toolchain::SmallVector<tfrt::AsyncValueRef<tfrt_stub::FallbackTensor>> outputs,
      gpu_runner_->Run(run_inputs));

  toolchain::SmallVector<tfrt::RCReference<tfrt::AsyncValue>, 4> outputs_to_wait;
  for (const auto& output : outputs) {
    if (!output.IsAvailable()) {
      outputs_to_wait.push_back(output.CopyRCRef());
    }
  }
  exec_ctx_->host()->Await(outputs_to_wait);

  ASSERT_EQ(outputs.size(), 1);
  auto expected = CreateTensor<int32>(TensorShape({1, 2}), {4, 6});
  test::ExpectTensorEqual<int32>(expected, outputs[0].get().tensor());
}

}  // namespace
}  // namespace gpu
}  // namespace machina
#endif
