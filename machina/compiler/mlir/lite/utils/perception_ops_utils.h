/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Saturday, May 24, 2025.
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

#ifndef MACHINA_COMPILER_MLIR_LITE_UTILS_PERCEPTION_OPS_UTILS_H_
#define MACHINA_COMPILER_MLIR_LITE_UTILS_PERCEPTION_OPS_UTILS_H_

#include <string>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/Attributes.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_attributes.h"

namespace mlir {
namespace TFL {

// Fuse MaxUnpooling2D ops annotated by tf.function to a TFLite custom op.
class ConvertMaxUnpoolingFunc {
 public:
  explicit ConvertMaxUnpoolingFunc(func::FuncOp func, mlir::TF::FuncAttr attr)
      : func_(func), attr_(attr) {}

  LogicalResult RewriteFunc();

  LogicalResult VerifySignature();

 private:
  LogicalResult CreateCustomOptions(std::string& custom_option_buffer);

  func::FuncOp func_;
  mlir::TF::FuncAttr attr_;
};

// Fuse DenseImageWarp ops annotated by tf.function to a TFLite custom op.
class ConvertDenseImageWarpFunc {
 public:
  explicit ConvertDenseImageWarpFunc(func::FuncOp func) : func_(func) {}

  LogicalResult RewriteFunc();

  LogicalResult VerifySignature();

 private:
  func::FuncOp func_;
};

}  // end namespace TFL
}  // end namespace mlir

#endif  // MACHINA_COMPILER_MLIR_LITE_UTILS_PERCEPTION_OPS_UTILS_H_
