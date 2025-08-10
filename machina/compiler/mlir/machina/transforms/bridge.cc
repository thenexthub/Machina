/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, May 15, 2025.
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

#include "machina/compiler/mlir/machina/transforms/bridge.h"

#include "absl/log/log.h"
#include "toolchain/ADT/StringRef.h"
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/Pass/PassManager.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/transforms/passes.h"
#include "machina/compiler/mlir/machina/utils/dump_mlir_util.h"
#include "machina/compiler/mlir/machina/utils/error_util.h"
#include "machina/compiler/mlir/tf2xla/internal/logging_hooks.h"
#include "machina/core/platform/status.h"
#include "machina/core/protobuf/core_platform_payloads.pb.h"

namespace mlir {
namespace TF {

absl::Status RunBridgeWithStandardPipeline(ModuleOp module, bool enable_logging,
                                           bool enable_inliner) {
  PassManager bridge(module.getContext());

  StandardPipelineOptions pipeline_options;
  pipeline_options.enable_inliner.setValue(enable_inliner);
  CreateTFStandardPipeline(bridge, pipeline_options);

  mlir::StatusScopedDiagnosticHandler diag_handler(
      module.getContext(), /*propagate=*/false,
      /*filter_stack=*/!VLOG_IS_ON(1));

  constexpr char kBridgeComponent[] = "TFXLABridge";
  if (enable_logging || VLOG_IS_ON(1)) {
    machina::DumpMlirOpToFile(kStandardPipelineBefore, module, "", &bridge);
    if (VLOG_IS_ON(2)) {
      machina::tf2xla::internal::EnablePassIRPrinting(bridge,
                                                         kBridgeComponent);
    }
  }
  LogicalResult result = bridge.run(module);
  (void)result;
  if (enable_logging || VLOG_IS_ON(1))
    machina::DumpMlirOpToFile(kStandardPipelineAfter, module, "", &bridge);
  return diag_handler.ConsumeStatus();
}

}  // namespace TF
}  // namespace mlir
