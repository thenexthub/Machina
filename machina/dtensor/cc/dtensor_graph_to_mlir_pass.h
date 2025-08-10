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

#ifndef MACHINA_DTENSOR_CC_DTENSOR_GRAPH_TO_MLIR_PASS_H_
#define MACHINA_DTENSOR_CC_DTENSOR_GRAPH_TO_MLIR_PASS_H_

#include <memory>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/Pass/PassManager.h"  // part of Codira Toolchain
#include "machina/core/common_runtime/device_set.h"
#include "machina/core/framework/function.h"
#include "machina/core/graph/graph.h"
#include "machina/core/platform/fingerprint.h"
#include "machina/dtensor/mlir/ir/tf_dtensor.h"

namespace machina {

class DTensorMlirPassRunner {
 public:
  DTensorMlirPassRunner();

  // Imports Graph to MLIR module in tf_execute Dialect with DTensor attributes.
  absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ImportGraphToMlir(
      const DeviceSet& device_set, absl::string_view name, bool is_func,
      const dtensor::Mesh& default_mesh,
      const FunctionLibraryDefinition& flib_def, const Graph& graph,
      Fprint128 cache_key);

  // Transforms input MLIR module with DTensor Pass pipeline.
  absl::Status Run(mlir::ModuleOp module);

 private:
  // N.B. op_registration_ must be initialized before context/pass-manager to
  // ensure DTensor operations are available during optimization passes.
  bool op_registration_ = mlir::TF::RegisterDTensorTFOps();
  mlir::MLIRContext context_;
  mlir::PassManager pass_manager_;

  bool logging_enabled_;
};

}  // namespace machina

#endif  // MACHINA_DTENSOR_CC_DTENSOR_GRAPH_TO_MLIR_PASS_H_
