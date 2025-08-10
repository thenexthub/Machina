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

#ifndef MACHINA_COMPILER_MLIR_LITE_FLATBUFFER_TRANSLATE_H_
#define MACHINA_COMPILER_MLIR_LITE_FLATBUFFER_TRANSLATE_H_

#include <string>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/op_or_arg_name_mapper.h"

namespace tflite {

// Translates the given MLIR `module` into a FlatBuffer and stores the
// serialized flatbuffer into the string. This uses OpOrArgLocNameMapper to
// convert location of the op to name in flatbuffer. Returns true if translation
// fails, otherwise returns false.
bool MlirToFlatBufferTranslateFunction(mlir::ModuleOp module,
                                       std::string* serialized_flatbuffer,
                                       bool emit_builtin_tflite_ops,
                                       bool emit_select_tf_ops,
                                       bool emit_custom_ops);

// Same as the above but with a custom op name mapper.
bool MlirToFlatBufferTranslateFunction(
    mlir::ModuleOp module, std::string* serialized_flatbuffer,
    bool emit_builtin_tflite_ops, bool emit_select_tf_ops, bool emit_custom_ops,
    machina::OpOrArgNameMapper* op_or_arg_name_mapper);
}  // namespace tflite

#endif  // MACHINA_COMPILER_MLIR_LITE_FLATBUFFER_TRANSLATE_H_
