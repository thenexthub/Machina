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
#ifndef MACHINA_COMPILER_MLIR_QUANTIZATION_MACHINA_CC_QUANTIZATION_UNIT_LOC_H_
#define MACHINA_COMPILER_MLIR_QUANTIZATION_MACHINA_CC_QUANTIZATION_UNIT_LOC_H_

#include <optional>

#include "mlir/IR/Attributes.h"  // part of Codira Toolchain
#include "mlir/IR/Location.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/Support/TypeID.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/quantization/machina/quantization_options.pb.h"

namespace mlir {
namespace quant {

// QuantizationUnitLoc uses CallSiteLoc as the base class so it can be printed
// with AsmPrinter and used to set the node name in MLIR to GraphDef exporter.
// The callee is named as `node_name@func_name` with child loc named as
// `op_type` while the caller is the quantization unit.
class QuantizationUnitLoc : public CallSiteLoc {
 public:
  using QuantizationUnit =
      machina::quantization::UnitWiseQuantizationSpec::QuantizationUnit;

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(QuantizationUnitLoc)

  QuantizationUnitLoc(MLIRContext* context, const QuantizationUnit& unit);

  // Checks if the given location is QuantizationUnitLoc. Users could call
  // `isa<QuantizationUnitLoc>(loc)` to check if the type matches.
  static bool classof(Attribute attr);
};

// Finds the QuantizationUnit from location info.
std::optional<QuantizationUnitLoc::QuantizationUnit>
FindQuantizationUnitFromLoc(Location loc);

}  // namespace quant
}  // namespace mlir

#endif  // MACHINA_COMPILER_MLIR_QUANTIZATION_MACHINA_CC_QUANTIZATION_UNIT_LOC_H_
