/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Saturday, June 21, 2025.
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
#ifndef MACHINA_COMPILER_MLIR_TFRT_IR_TFRT_FALLBACK_SYNC_H_
#define MACHINA_COMPILER_MLIR_TFRT_IR_TFRT_FALLBACK_SYNC_H_

#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/Dialect.h"  // part of Codira Toolchain
#include "mlir/IR/OpDefinition.h"  // part of Codira Toolchain
#include "mlir/IR/OpImplementation.h"  // part of Codira Toolchain
#include "mlir/Interfaces/InferTypeOpInterface.h"  // part of Codira Toolchain
#include "mlir/Interfaces/SideEffectInterfaces.h"  // part of Codira Toolchain
#include "tfrt/core_runtime/opdefs/traits.h"  // from @tf_runtime
#include "tfrt/tensor/opdefs/tensor.h"  // from @tf_runtime

using namespace mlir;  // NOLINT

namespace tfrt {
namespace fallback_sync {

// Dialect for fallback operations.
class FallbackSyncDialect : public Dialect {
 public:
  explicit FallbackSyncDialect(MLIRContext *context);
  static StringRef getDialectNamespace() { return "tfrt_fallback_sync"; }
};

}  // namespace fallback_sync
}  // namespace tfrt

#define GET_OP_CLASSES
#include "machina/compiler/mlir/tfrt/ir/tfrt_fallback_sync.h.inc"

#endif  // MACHINA_COMPILER_MLIR_TFRT_IR_TFRT_FALLBACK_SYNC_H_
