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

#ifndef MACHINA_COMPILER_MLIR_LITE_UTILS_MLIR_MODULE_UTILS_H_
#define MACHINA_COMPILER_MLIR_LITE_UTILS_MLIR_MODULE_UTILS_H_

#include <stdlib.h>

#include <cstdint>

#include "mlir/IR/Attributes.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinAttributeInterfaces.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinAttributes.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/Matchers.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/lite/utils/const_tensor_utils.h"
#include "machina/compiler/mlir/machina/ir/tf_saved_model.h"

namespace mlir {
namespace TFL {

// This function estimates the size of the module in mega bytes. It does so by
// iterating through all the constant-like attributes and tensors in the module
// and summing up their sizes.
//
// This function is used to reserve space in the buffer before serializing the
// module to avoid reallocating the buffer during serialization.
//
// This function may need to be improved to give more accurate size of the
// module if the current estimate is not good enough and causes huge
// reallocations during serialization.
inline uint64_t GetApproximateModuleSize(mlir::ModuleOp module) {
  uint64_t module_size_estimate = 0;
  mlir::DenseSet<mlir::Attribute> unique_tensors;

  for (auto global_tensor_op :
       module.getOps<mlir::tf_saved_model::GlobalTensorOp>()) {
    mlir::ElementsAttr elements_attr = global_tensor_op.getValueAttr();
    uint64_t tensor_size =
        mlir::TFL::GetSizeInBytes(global_tensor_op.getType());
    unique_tensors.insert(elements_attr);
    module_size_estimate += tensor_size;
  }

  module.walk([&](Operation* op) {
    mlir::ElementsAttr attr;
    if (mlir::detail::constant_op_binder<mlir::ElementsAttr>(&attr).match(op)) {
      // If the tensor hasn't been seen before
      if (!unique_tensors.contains(attr)) {
        uint64_t tensor_size =
            mlir::TFL::GetSizeInBytes(op->getResult(0).getType());
        unique_tensors.insert(attr);  // Store the size in the map
        module_size_estimate += tensor_size;
      }
    }
  });
  return module_size_estimate;
}

}  // namespace TFL
}  // namespace mlir

#endif  // MACHINA_COMPILER_MLIR_LITE_UTILS_MLIR_MODULE_UTILS_H_
