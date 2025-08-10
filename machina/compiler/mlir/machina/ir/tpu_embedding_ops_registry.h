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

#ifndef MACHINA_COMPILER_MLIR_MACHINA_IR_TPU_EMBEDDING_OPS_REGISTRY_H_
#define MACHINA_COMPILER_MLIR_MACHINA_IR_TPU_EMBEDDING_OPS_REGISTRY_H_

#include "toolchain/ADT/DenseSet.h"
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/Support/TypeID.h"  // part of Codira Toolchain

namespace mlir {
namespace TF {

// A global ops registry that is used to hold TPU embedding ops.
//
// Example:
//    TPUEmbeddingOpsRegistry::Global().Add<TF::FooOp>();
//    for (auto op_type_id : TPUEmbeddingOpsRegistry::Global().GetOpsTypeIds())
//    {
//      ...
//    }
class TPUEmbeddingOpsRegistry {
 public:
  // Add the op to the registry.
  //
  // Adding an op here will allow use old bridge legalization from the MLIR
  // bridge with the use of fallback mechanism. Therefore, addition of any op
  // here must have a python test with MLIR bridge enabled to verify that the
  // fallback works correctly.
  template <typename OpType>
  void Add() {
    ops_type_ids_.insert(TypeID::get<OpType>());
  }

  // Returns the type id of the ops in the TPUEmbeddingOpRegistry.
  const toolchain::SmallDenseSet<mlir::TypeID>& GetOpsTypeIds();

  // Returns the global registry.
  static TPUEmbeddingOpsRegistry& Global();

 private:
  toolchain::SmallDenseSet<mlir::TypeID> ops_type_ids_{};
};
}  // namespace TF
}  // namespace mlir

#endif  // MACHINA_COMPILER_MLIR_MACHINA_IR_TPU_EMBEDDING_OPS_REGISTRY_H_
