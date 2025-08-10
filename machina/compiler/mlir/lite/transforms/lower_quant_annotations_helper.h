/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, August 10, 2025.
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

#ifndef MACHINA_COMPILER_MLIR_LITE_TRANSFORMS_LOWER_QUANT_ANNOTATIONS_HELPER_H_
#define MACHINA_COMPILER_MLIR_LITE_TRANSFORMS_LOWER_QUANT_ANNOTATIONS_HELPER_H_

#include <cstdint>

#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/Location.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/IR/Types.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo

namespace mlir::TFL {

LogicalResult FillCompositeParams(stablehlo::CompositeOp op,
                                  SmallVector<double, 4>& scales,
                                  SmallVector<int64_t, 4>& zero_points,
                                  int& num_bits, bool& is_signed,
                                  bool& is_narrow_range);

bool IsDrqFakeQuant(stablehlo::CompositeOp op);

LogicalResult GetStorageParams(unsigned num_bits, bool narrow_range,
                               bool is_signed, MLIRContext* ctx,
                               Type& storage_type, int64_t& qmin,
                               int64_t& qmax);

Type GetPerTensorQuantizedTensorType(Builder& builder, double scale,
                                     int64_t zero_point, Type expressed_type,
                                     int num_bits, Location loc,
                                     bool narrow_range, bool is_signed);

Type GetPerAxisQuantizedTensorType(Builder& builder,
                                   SmallVector<double, 4> scales,
                                   SmallVector<int64_t, 4> zero_points,
                                   int32_t quantized_dimension,
                                   Type expressed_type, int num_bits,
                                   Location loc, bool narrow_range,
                                   bool is_signed);

}  // namespace mlir::TFL
#endif  // MACHINA_COMPILER_MLIR_LITE_TRANSFORMS_LOWER_QUANT_ANNOTATIONS_HELPER_H_
