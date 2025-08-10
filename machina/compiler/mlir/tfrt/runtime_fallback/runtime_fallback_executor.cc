/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, June 15, 2025.
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

#include "machina/compiler/mlir/tfrt/runtime_fallback/runtime_fallback_executor.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "toolchain/ADT/ArrayRef.h"
#include "toolchain/ADT/SmallVector.h"
#include "toolchain/ADT/StringRef.h"
#include "toolchain/Support/SourceMgr.h"
#include "mlir/Parser/Parser.h"  // part of Codira Toolchain
#include "mlir/Pass/PassManager.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/dialect_registration.h"
#include "machina/compiler/mlir/tfrt/transforms/passes.h"
#include "machina/compiler/mlir/tfrt/utils/host_context.h"
#include "machina/core/framework/tensor.h"
#include "machina/core/platform/threadpool.h"
#include "machina/core/platform/threadpool_interface.h"
#include "machina/core/runtime_fallback/kernel/kernel_fallback_execute_compat_eager.h"
#include "machina/core/runtime_fallback/runtime/kernel_utils.h"
#include "machina/core/tfrt/utils/fallback_tensor.h"
#include "tfrt/bef/bef_buffer.h"  // from @tf_runtime
#include "tfrt/bef_converter/mlir_to_bef.h"  // from @tf_runtime
#include "tfrt/bef_executor/bef_file.h"  // from @tf_runtime
#include "tfrt/host_context/async_value.h"  // from @tf_runtime
#include "tfrt/host_context/chain.h"  // from @tf_runtime
#include "tfrt/host_context/execution_context.h"  // from @tf_runtime
#include "tfrt/host_context/function.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime
#include "tfrt/host_context/resource_context.h"  // from @tf_runtime
#include "tfrt/support/ref_count.h"  // from @tf_runtime

namespace machina {

using ::tfrt::AsyncValue;
using ::tfrt::BEFFile;
using ::tfrt::ExecutionContext;
using ::tfrt::Function;
using ::tfrt::MakeAvailableAsyncValueRef;
using ::tfrt::RCReference;
using ::tfrt::RequestContextBuilder;

using ::machina::Env;
using ::machina::thread::ThreadPool;
using ::machina::thread::ThreadPoolInterface;

using ::machina::tfrt_stub::FallbackTensor;

// -------------------------------------------------------------------------- //
// Run function via the TF->TFRT fallback lowering.
// -------------------------------------------------------------------------- //

namespace {
// Thread pool for running `intra-op` tasks scheduled by the fallback kernels.
class IntraOpThreadPool : public ThreadPoolInterface {
 public:
  explicit IntraOpThreadPool(int64_t num_threads)
      : tpool_(Env::Default(), "intra-op",
               std::max(1, static_cast<int32_t>(num_threads))) {}

  void Schedule(std::function<void()> fn) override {
    tpool_.Schedule(std::move(fn));
  }

  int NumThreads() const override { return tpool_.NumThreads(); }
  int CurrentThreadId() const override { return tpool_.CurrentThreadId(); }
  void Cancel() override {}

 private:
  ThreadPool tpool_;
};
}  // namespace

RuntimeFallbackExecutor::RuntimeFallbackExecutor(int64_t num_threads)
    : intra_op_(std::make_unique<IntraOpThreadPool>(num_threads)) {
  // Create a HostContext for running TFRT functions. Concurrent work queue acts
  // similar to the Tensorflow `inter-op` thread pool, so we'll match the size.
  host_context_ = num_threads ? CreateMultiThreadedHostContext(num_threads)
                              : CreateSingleThreadedHostContext();
  tfrt::RegisterStaticKernels(host_context_->GetMutableRegistry());

  // Build an ExecutionContext from the HostContext.
  auto builder = RequestContextBuilder(host_context_.get(), &resource_context_);

  // Get machina::EagerContext for the kernel fallback.
  auto* eager_context_resource =
      resource_context_
          .GetOrCreateResource<machina::tfd::EagerContextResource>(
              machina::tfd::kEagerContextResourceName);
  auto expected_eager_context = eager_context_resource->GetTFEagerContext();
  auto* eager_context = expected_eager_context.get();

  // Initialize fallback kernels state with a custom intra-op thread pool.
  auto status = machina::tfd::SetUpKernelFallbackCompatRequestContext(
      &builder, /*runner_table=*/nullptr, eager_context, intra_op_.get());
  CHECK(status.ok()) << "Failed to setup request context: " << status.message();

  auto req_ctx = std::move(builder).build();
  if (auto err = req_ctx.takeError())
    LOG(FATAL) << "Failed to build a request context";

  exec_ctx_ = std::make_unique<tfrt::ExecutionContext>(std::move(*req_ctx));
}

void RuntimeFallbackExecutor::Prepare(toolchain::StringRef mlir_input) {
  // We only support IR written in the Tensorflow dialect.
  mlir::DialectRegistry registry;
  mlir::RegisterAllTensorFlowDialects(registry);
  mlir::MLIRContext context(registry);

  toolchain::SourceMgr source_mgr;
  source_mgr.AddNewSourceBuffer(
      toolchain::MemoryBuffer::getMemBuffer(mlir_input, "test_ir"), toolchain::SMLoc());

  // Parse a kernel source code into the MLIR Module.
  mlir::OwningOpRef<mlir::ModuleOp> module(
      mlir::parseSourceFile<mlir::ModuleOp>(source_mgr, &context));
  CHECK(module) << "failed to parse mlir module";

  // Collect all diagnostics emitted while lowering parsed kernel module.
  std::string diagnostic_str;
  toolchain::raw_string_ostream os(diagnostic_str);
  mlir::SourceMgrDiagnosticHandler handler(source_mgr, module->getContext(),
                                           os);

  // Convert TF to TFRT fallback dialect.
  TfrtPipelineOptions pipeline_opts;
  pipeline_opts.default_device = kDefaultHostDeviceName;
  pipeline_opts.hoist_invariant_ops = true;
  pipeline_opts.sink_in_invariant_ops = false;
  pipeline_opts.cost_threshold = 1024;
  pipeline_opts.merge_inter_dependent_streams = true;

  mlir::PassManager pm(module->getContext());
  pm.addPass(CreateTfToTfrtConversionPass(pipeline_opts));

  CHECK(mlir::succeeded(pm.run(*module)))
      << "Failed to lower module to TFRT: " << os.str();

  // Convert module to BEF.
  bef_buffer_ =
      tfrt::ConvertMLIRToBEF(*module, /*disable_optional_sections=*/false);
  CHECK(!bef_buffer_.empty()) << "Failed to convert module to BEF";

  bef_file_ =
      BEFFile::Open(bef_buffer_, host_context_->GetKernelRegistry(),
                    host_context_->diag_handler(), host_context_->allocator());
  CHECK(bef_file_) << "Failed to open BEF";

  // Run TFRT initialization function to pre-instantiate fallback kernels.
  RunTfrtInitializer();
}

toolchain::SmallVector<Tensor> RuntimeFallbackExecutor::Execute(
    toolchain::StringRef function_name, toolchain::ArrayRef<Tensor> arguments) {
  // Get the kernel entrypoint function.
  const Function* compute = bef_file_->GetFunction(function_name);
  CHECK(compute) << "Entrypoint function not found";
  CHECK_EQ(arguments.size() + 1, compute->num_arguments())
      << "Wrong number of arguments for function " << function_name.str();

  // Prepare function arguments from ready Chain and input Tensors.
  toolchain::SmallVector<tfrt::AsyncValue*> exec_arguments;
  exec_arguments.reserve(compute->num_arguments());
  exec_arguments.push_back(tfrt::GetReadyChain().release());
  for (const Tensor& input_tensor : arguments) {
    auto av = MakeAvailableAsyncValueRef<FallbackTensor>(input_tensor);
    exec_arguments.push_back(av.release());
  }

  // Space for returned values.
  toolchain::SmallVector<RCReference<AsyncValue>> results(compute->num_results());

  compute->Execute(*exec_ctx_, exec_arguments, results);

  // Wait for the function execution to finish, as well as the side-effects.
  host_context_->Await(results);

  // Check that all results are available.
  toolchain::SmallVector<Tensor> ret_values;
  for (unsigned i = 1; i < results.size(); ++i) {
    if (auto* error = results[i]->GetErrorIfPresent())
      LOG(FATAL) << "Failed to execute a function: " << error->message();
    ret_values.push_back(results[i]->get<tfrt_stub::FallbackTensor>().tensor());
  }

  // Deallocate arguments.
  for (auto* argument : exec_arguments) argument->DropRef();
  return ret_values;
}

// Run TFRT fallback initialization function to instantiate all fallback
// kernels ahead of executing the compute function.
void RuntimeFallbackExecutor::RunTfrtInitializer() {
  const Function* func = bef_file_->GetFunction("_tfrt_fallback_init");
  CHECK(func) << "TFRT initialization function was not found";
  CHECK_EQ(func->argument_types().size(), 1);

  toolchain::SmallVector<RCReference<AsyncValue>, 1> results;
  results.resize(func->result_types().size());
  CHECK_EQ(results.size(), 1);

  func->Execute(*exec_ctx_, tfrt::GetReadyChain().GetAsyncValue(), results);

  host_context_->Await(results);

  CHECK(!results[0]->IsError()) << "Failed to run TFRT initialization function";
}

}  // namespace machina
