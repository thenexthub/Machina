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
#include "machina/compiler/mlir/tfrt/ir/mlrt/tf_mlrt_tpu_ops.h"

#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypes.h"  // part of Codira Toolchain
#include "mlir/IR/DialectImplementation.h"  // part of Codira Toolchain
#include "mlir/IR/OpDefinition.h"  // part of Codira Toolchain
#include "mlir/IR/TypeUtilities.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_ops.h"
#include "machina/compiler/mlir/tfrt/ir/mlrt/mlrt_dialect.h"
#include "machina/compiler/mlir/tfrt/ir/mlrt/tf_mlrt_ops.h"

namespace machina {
namespace tf_mlrt_tpu {

TensorflowMlrtTpuDialect::TensorflowMlrtTpuDialect(mlir::MLIRContext *context)
    : mlir::Dialect(/*name=*/"tf_mlrt_tpu", context,
                    mlir::TypeID::get<TensorflowMlrtTpuDialect>()) {
  addOperations<
#define GET_OP_LIST
#include "machina/compiler/mlir/tfrt/ir/mlrt/tf_mlrt_tpu_ops.cpp.inc"
      >();
}

}  // namespace tf_mlrt_tpu
}  // namespace machina

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "machina/compiler/mlir/tfrt/ir/mlrt/tf_mlrt_tpu_ops.cpp.inc"
