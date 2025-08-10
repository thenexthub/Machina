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

// This file defines the operations used in the Runtime Fallback dialect.

#ifndef MACHINA_COMPILER_MLIR_TFRT_RUNTIME_FALLBACK_RUNTIME_FALLBACK_OPS_H_
#define MACHINA_COMPILER_MLIR_TFRT_RUNTIME_FALLBACK_RUNTIME_FALLBACK_OPS_H_

#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/Dialect.h"  // part of Codira Toolchain
#include "mlir/IR/OpDefinition.h"  // part of Codira Toolchain
#include "mlir/IR/TypeUtilities.h"  // part of Codira Toolchain
#include "mlir/Interfaces/SideEffectInterfaces.h"  // part of Codira Toolchain
#include "tfrt/tensor/opdefs/tensor.h"  // from @tf_runtime

namespace mlir {
namespace tfd {

// Dialect for TFRT delegate operations.
class RuntimeFallbackDialect : public Dialect {
 public:
  explicit RuntimeFallbackDialect(MLIRContext* context);
  static StringRef getDialectNamespace() { return "tfd"; }
};

}  // namespace tfd
}  // namespace mlir

#define GET_OP_CLASSES
#include "machina/compiler/mlir/tfrt/runtime_fallback_ops.h.inc"

#endif  // MACHINA_COMPILER_MLIR_TFRT_RUNTIME_FALLBACK_RUNTIME_FALLBACK_OPS_H_
