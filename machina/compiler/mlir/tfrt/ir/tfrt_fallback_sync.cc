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
#include "machina/compiler/mlir/tfrt/ir/tfrt_fallback_sync.h"

#include "toolchain/ADT/STLExtras.h"
#include "mlir/IR/Attributes.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypes.h"  // part of Codira Toolchain
#include "mlir/IR/DialectImplementation.h"  // part of Codira Toolchain
#include "mlir/IR/OpDefinition.h"  // part of Codira Toolchain
#include "mlir/IR/OpImplementation.h"  // part of Codira Toolchain
#include "mlir/IR/OperationSupport.h"  // part of Codira Toolchain
#include "mlir/IR/TypeUtilities.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/tfrt/ir/tfrt_fallback.h"
#include "machina/compiler/mlir/tfrt/ir/tfrt_fallback_common.h"
#include "tfrt/basic_kernels/opdefs/tfrt_base.h"  // from @tf_runtime
#include "tfrt/basic_kernels/opdefs/types.h"  // from @tf_runtime
#include "tfrt/core_runtime/opdefs/attributes.h"  // from @tf_runtime
#include "tfrt/core_runtime/opdefs/core_runtime.h"  // from @tf_runtime
#include "tfrt/core_runtime/opdefs/types.h"  // from @tf_runtime
#include "tfrt/tensor/opdefs/tensor.h"  // from @tf_runtime

namespace tfrt {
namespace fallback_sync {

FallbackSyncDialect::FallbackSyncDialect(MLIRContext *context)
    : Dialect(/*name=*/"tfrt_fallback_sync", context,
              TypeID::get<FallbackSyncDialect>()) {
  context->getOrLoadDialect<tfrt::fallback::FallbackDialect>();
  context->getOrLoadDialect<compiler::TFRTDialect>();
  context->getOrLoadDialect<corert::CoreRTDialect>();

  allowUnknownTypes();

  addOperations<
#define GET_OP_LIST
#include "machina/compiler/mlir/tfrt/ir/tfrt_fallback_sync.cpp.inc"
      >();
}

static Type GetTensorType(Builder *builder) {
  return tfrt::tfrt_tensor::TensorType::get(builder->getContext());
}

}  // namespace fallback_sync
}  // namespace tfrt

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "machina/compiler/mlir/tfrt/ir/tfrt_fallback_sync.cpp.inc"
