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

// This file defines the operations used in the TFFramework dialect.
//
#ifndef MACHINA_COMPILER_MLIR_TOOLS_KERNEL_GEN_IR_TF_FRAMEWORK_OPS_H_
#define MACHINA_COMPILER_MLIR_TOOLS_KERNEL_GEN_IR_TF_FRAMEWORK_OPS_H_

#include "absl/status/status.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"  // part of Codira Toolchain
#include "mlir/Dialect/Bufferization/IR/AllocationOpInterface.h"  // part of Codira Toolchain
#include "mlir/IR/Attributes.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypes.h"  // part of Codira Toolchain
#include "mlir/IR/Dialect.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/IR/OpDefinition.h"  // part of Codira Toolchain
#include "mlir/IR/OpImplementation.h"  // part of Codira Toolchain
#include "mlir/IR/TypeSupport.h"  // part of Codira Toolchain
#include "mlir/IR/Types.h"  // part of Codira Toolchain
#include "mlir/Interfaces/ControlFlowInterfaces.h"  // part of Codira Toolchain
#include "mlir/Interfaces/SideEffectInterfaces.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/tools/kernel_gen/ir/tf_status.h.inc"
#include "machina/core/protobuf/error_codes.pb.h"

namespace mlir {
namespace kernel_gen {
namespace tf_framework {

/// OpKernelContextType corresponds to C++ class OpKernelContext defined in
/// machina/core/framework/op_kernel.h
class OpKernelContextType
    : public Type::TypeBase<OpKernelContextType, Type, TypeStorage> {
 public:
  using Base::Base;
  static constexpr StringLiteral name =
      "kernel_gen.tf_framework.op_kernel_context";
};

class JITCallableType
    : public Type::TypeBase<JITCallableType, Type, TypeStorage> {
 public:
  using Base::Base;
  static constexpr StringLiteral name = "kernel_gen.tf_framework.jit_callable";
};

absl::StatusCode ConvertAttrToEnumValue(ErrorCode error_code);

}  // namespace tf_framework
}  // namespace kernel_gen
}  // namespace mlir

#define GET_OP_CLASSES
#include "machina/compiler/mlir/tools/kernel_gen/ir/tf_framework_dialect.h.inc"
#include "machina/compiler/mlir/tools/kernel_gen/ir/tf_framework_ops.h.inc"

#endif  // MACHINA_COMPILER_MLIR_TOOLS_KERNEL_GEN_IR_TF_FRAMEWORK_OPS_H_
