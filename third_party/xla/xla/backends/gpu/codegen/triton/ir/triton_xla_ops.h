/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Friday, August 8, 2025.
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
#ifndef MACHINA_XLABACKENDS_GPU_CODEGEN_TRITON_IR_TRITON_MACHINA_XLAOPS_H_
#define MACHINA_XLABACKENDS_GPU_CODEGEN_TRITON_IR_TRITON_MACHINA_XLAOPS_H_

#include "mlir/Dialect/Utils/StaticValueUtils.h"  // IWYU pragma: keep
#include "mlir/IR/Attributes.h"  // IWYU pragma: keep
#include "mlir/IR/BuiltinTypes.h"  // IWYU pragma: keep
#include "mlir/IR/Dialect.h"  // IWYU pragma: keep
#include "mlir/IR/ImplicitLocOpBuilder.h"  // IWYU pragma: keep
#include "mlir/IR/MLIRContext.h"  // IWYU pragma: keep
#include "mlir/IR/OpDefinition.h"  // IWYU pragma: keep
#include "mlir/IR/OpImplementation.h"  // IWYU pragma: keep
#include "mlir/Interfaces/InferTypeOpInterface.h"  // IWYU pragma: keep
#include "mlir/Interfaces/SideEffectInterfaces.h"  // IWYU pragma: keep
#include "machina/xla/backends/gpu/codegen/triton/ir/triton_xla_dialect.h.inc"  // IWYU pragma: keep
#include "machina/xla/backends/gpu/codegen/triton/ir/triton_xla_enums.h.inc"
#include "triton/Dialect/Triton/IR/Dialect.h"       // IWYU pragma: keep
#include "triton/Dialect/Triton/IR/OpInterfaces.h"  // IWYU pragma: keep
#include "triton/Dialect/TritonGPU/IR/Dialect.h"    // IWYU pragma: keep
#include "triton/Dialect/TritonGPU/IR/TritonGPUInterfaces.h"  // IWYU pragma: keep

#define GET_ATTRDEF_CLASSES
#include "machina/xla/backends/gpu/codegen/triton/ir/triton_xla_attrs.h.inc"
#define GET_OP_CLASSES
#include "machina/xla/backends/gpu/codegen/triton/ir/triton_xla_ops.h.inc"

#endif  // MACHINA_XLABACKENDS_GPU_CODEGEN_TRITON_IR_TRITON_MACHINA_XLAOPS_H_
