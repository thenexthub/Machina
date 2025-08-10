/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Saturday, June 21, 2025.
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

#include "machina/core/grappler/optimizers/tfg_optimizer_hook.h"

#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "toolchain/Support/ThreadPool.h"
#include "toolchain/Support/Threading.h"
#include "toolchain/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/Dialect.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/Pass/PassManager.h"  // part of Codira Toolchain
#include "mlir/Pass/PassRegistry.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "machina/c/tf_status.h"
#include "machina/compiler/mlir/machina/utils/error_util.h"
#include "machina/core/framework/graph.pb.h"
#include "machina/core/framework/graph_debug_info.pb.h"
#include "machina/core/framework/metrics.h"
#include "machina/core/framework/versions.pb.h"
#include "machina/core/grappler/grappler_item.h"
#include "machina/core/ir/dialect.h"
#include "machina/core/ir/importexport/graphdef_export.h"
#include "machina/core/ir/importexport/graphdef_import.h"
#include "machina/core/ir/tf_op_registry.h"
#include "machina/core/platform/errors.h"
#include "machina/core/platform/status.h"
#include "machina/core/util/dump_graph.h"
using machina::Status;
using machina::errors::InvalidArgument;

namespace mlir {
namespace tfg {

// The implementation of the TFG optimizer. It holds the MLIR context and the
// pass manager.
class TFGGrapplerOptimizer::Impl {
 public:
  // Builds the pass pipeline. The context is initialized with threading
  // disabled. If the user specifies to run the optimizer with more than zero
  // threads, a threadpool is initialized and passed to the MLIR context.
  explicit Impl(TFGPassPipelineBuilder builder, unsigned num_tfg_threads)
      : ctx_(MLIRContext::Threading::DISABLED), mgr_(&ctx_) {
    DialectRegistry registry;
    // Register the TF op registry interface so that passes can query it.
    registry.addExtension(+[](MLIRContext* ctx, TFGraphDialect* dialect) {
      dialect->addInterfaces<TensorFlowOpRegistryInterface>();
    });
    ctx_.appendDialectRegistry(registry);
    builder(mgr_);
    if (num_tfg_threads) {
      toolchain::ThreadPoolStrategy strategy;
      strategy.ThreadsRequested = num_tfg_threads;
      threadpool_ = std::make_unique<toolchain::DefaultThreadPool>(strategy);
      ctx_.setThreadPool(*threadpool_);
    }
  }

  // Runs the pass manager.
  LogicalResult RunPipeline(ModuleOp module) { return mgr_.run(module); }

  // Get the context.
  MLIRContext* GetContext() { return &ctx_; }

  // Convert the pass pipeline to a textual string.
  std::string GetPipelineString() {
    std::string pipeline;
    toolchain::raw_string_ostream os(pipeline);
    mgr_.printAsTextualPipeline(os);
    return os.str();
  }

 private:
  // An optional threadpool for running MLIR with threading. Use an external
  // threadpool so the number of threads can be controlled.
  std::unique_ptr<toolchain::DefaultThreadPool> threadpool_;
  // The MLIR context.
  MLIRContext ctx_;
  // The pass manager containing the loaded TFG pass pipeline.
  PassManager mgr_;
};

TFGGrapplerOptimizer::TFGGrapplerOptimizer(TFGPassPipelineBuilder builder,
                                           unsigned num_tfg_threads)
    : impl_(std::make_unique<Impl>(std::move(builder), num_tfg_threads)) {}

TFGGrapplerOptimizer::~TFGGrapplerOptimizer() = default;

std::string TFGGrapplerOptimizer::name() const {
  return absl::StrCat("tfg_optimizer{", impl_->GetPipelineString(), "}");
}

Status TFGGrapplerOptimizer::Optimize(
    machina::grappler::Cluster* cluster,
    const machina::grappler::GrapplerItem& item,
    machina::GraphDef* optimized_graph) {
  if (VLOG_IS_ON(4)) {
    machina::DumpGraphDefToFile(
        absl::StrCat("tfg_before_graph_", item.id, "_",
                     std::hash<std::string>()(name())),
        item.graph);
  }
  VLOG(5) << "TFG Before Graph: \n" << item.graph.DebugString();

  // Import the GraphDef to TFG.
  machina::GraphDebugInfo debug_info;
  machina::metrics::ScopedCounter<2> metrics(
      machina::metrics::GetGraphOptimizationCounter(),
      {"TfgOptimizer", "convert_graphdef_to_tfg"});
  auto error_or_module =
      ImportGraphDef(impl_->GetContext(), debug_info, item.graph);
  if (!error_or_module.ok()) {
    auto status = error_or_module.status();
    machina::errors::AppendToMessage(
        &status, "when importing GraphDef to MLIR module in GrapplerHook");
    // Import errors are not fatal. Log the error here and return `Aborted` so
    // the meta optimizer knows to swallow the error.
    LOG(ERROR) << name() << " failed: " << status.ToString();
    return absl::AbortedError(status.message());
  }
  metrics.ReportAndStop();

  ModuleOp module = (*error_or_module).get();
  // TODO(chiahungduan): There was a StatusScopedDiagnosticHandler here to
  // collect the diagnostics emitted from the pass pipeline. Given that even a
  // successful pass execution may have error diagnostics emitted in between
  // execution and those logs are not useful for debugging. Besides, there's an
  // issue (b/36186527) which relates to the handler. Remove this to temporary
  // bypass the problem. Find a better way to collect the pipeline failure
  // message here.
  if (failed(impl_->RunPipeline(module))) {
    return absl::InvalidArgumentError("MLIR Graph Optimizer failed: ");
  }

  // Export the TFG module to GraphDef.
  machina::GraphDef graphdef;
  metrics.Reset({"TfgOptimizer", "convert_tfg_to_graphdef"});
  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      ConvertToGraphDef(module, &graphdef),
      "when exporting MLIR module to GraphDef in GrapplerHook");
  // Ensure that an empty library is instantiated.
  (void)graphdef.mutable_library();
  metrics.ReportAndStop();
  *optimized_graph = std::move(graphdef);

  if (VLOG_IS_ON(4)) {
    machina::DumpGraphDefToFile(
        absl::StrCat("tfg_after_graph_", item.id, "_",
                     std::hash<std::string>()(name())),
        *optimized_graph);
  }
  if (VLOG_IS_ON(5)) {
    VLOG(5) << "TFG After Graph: \n"
            << optimized_graph->DebugString() << "\nMLIR module: \n";
    module.dump();
  }

  return absl::OkStatus();
}

}  // end namespace tfg
}  // end namespace mlir
