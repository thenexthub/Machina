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
#ifndef MACHINA_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_COMPONENT_H_
#define MACHINA_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_COMPONENT_H_

#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/quantization/stablehlo/quantization_config.pb.h"

namespace mlir::quant::stablehlo {

// Component is a public abstraction for StableHLO Quantizer that represents the
// most basic unit of action applied to the StableHLO graph. Derived classes
// should override the `Run` method to implement the action.
class Component {
 public:
  virtual ~Component() = default;

  // Runs the action to the StableHLO graph, passed by the `module_op`. `config`
  // should provide information necessary to configure the action's behavior.
  virtual absl::StatusOr<ModuleOp> Run(
      ModuleOp module_op,
      const ::stablehlo::quantization::QuantizationConfig& config) = 0;
};

}  // namespace mlir::quant::stablehlo

#endif  // MACHINA_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_COMPONENT_H_
