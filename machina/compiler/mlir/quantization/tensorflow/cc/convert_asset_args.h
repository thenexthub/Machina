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
#ifndef MACHINA_COMPILER_MLIR_QUANTIZATION_MACHINA_CC_CONVERT_ASSET_ARGS_H_
#define MACHINA_COMPILER_MLIR_QUANTIZATION_MACHINA_CC_CONVERT_ASSET_ARGS_H_

#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "machina/core/protobuf/meta_graph.pb.h"

namespace mlir::quant {

// Converts arguments of the @main function that are bound to
// `tf_saved_model::AssetOp`s into regular tensor args. Returns `AsestFileDef`s
// that associates the arg with the asset.
//
// In detail, this function performs the following:
// * Replaces "tf_saved_model.bound_input" attributes to
//   "tf_saved_model.index_path", if the bound input is attached to the
//   `tf_saved_model::AssetOp`.
// * Strips the "assets/" prefix of the filename when setting it to
//   `AssetFileDef`.
FailureOr<SmallVector<machina::AssetFileDef>> ConvertAssetArgs(
    ModuleOp module_op);

}  // namespace mlir::quant

#endif  // MACHINA_COMPILER_MLIR_QUANTIZATION_MACHINA_CC_CONVERT_ASSET_ARGS_H_
