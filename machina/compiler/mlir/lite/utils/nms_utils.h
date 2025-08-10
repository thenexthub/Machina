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

// This header file defines common utils used by TFLite transformation
// passes to work with NMS ops in TFLite.

#ifndef MACHINA_COMPILER_MLIR_LITE_UTILS_NMS_UTILS_H_
#define MACHINA_COMPILER_MLIR_LITE_UTILS_NMS_UTILS_H_

#include <string>

#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/Attributes.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinAttributes.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_attributes.h"

namespace mlir {
namespace TFL {

// Abstracts the conversion of the padded NMS composite function.
class ConvertNMSPaddedFunc {
 public:
  explicit ConvertNMSPaddedFunc(func::FuncOp func) : func_(func) {}

  void RewriteFunc();

  LogicalResult VerifySignature();

 private:
  func::FuncOp func_;
};

// Abstracts the conversion of the SSD post-processing composite function to
// TFLite.
class ConvertSSDPostProcessFunc {
 public:
  explicit ConvertSSDPostProcessFunc(func::FuncOp func, mlir::TF::FuncAttr attr)
      : func_(func), attr_(attr) {}

  LogicalResult RewriteFunc();

  LogicalResult VerifySignature();

 private:
  LogicalResult CreateNMSCustomOptions(func::FuncOp func, DictionaryAttr attrs,
                                       std::string& custom_option_buffer);

  LogicalResult AddIntAttr(func::FuncOp func, DictionaryAttr attrs,
                           const std::string& attribute,
                           flexbuffers::Builder* builder);

  LogicalResult AddFloatAttr(func::FuncOp func, DictionaryAttr attrs,
                             const std::string& attribute,
                             flexbuffers::Builder* builder);

  LogicalResult HasIntAttr(func::FuncOp func, DictionaryAttr attrs,
                           const std::string& attribute);

  LogicalResult HasFloatAttr(func::FuncOp func, DictionaryAttr attrs,
                             const std::string& attribute);

  func::FuncOp func_;
  mlir::TF::FuncAttr attr_;
};

}  // end namespace TFL
}  // end namespace mlir

#endif  // MACHINA_COMPILER_MLIR_LITE_UTILS_TFTEXT_UTILS_H_
