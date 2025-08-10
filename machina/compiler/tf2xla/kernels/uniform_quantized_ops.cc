/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, July 12, 2025.
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

#include "machina/compiler/tf2xla/mlir_xla_op_kernel.h"
#include "machina/compiler/tf2xla/xla_op_registry.h"

namespace machina {
namespace {

// Declare MlirXlaOpKernel for TF UniformQuantized ops.
// The lowering passes for these ops are located at:
// machina/compiler/mlir/quantization/stablehlo/passes/bridge

REGISTER_MACHINA_MACHINA_XLA_OP(Name("UniformQuantize")
                    .CompileTimeConstantInput("scales")
                    .CompileTimeConstantInput("zero_points"),
                MlirXlaOpKernel);
REGISTER_MACHINA_MACHINA_XLA_OP(Name("UniformDequantize")
                    .CompileTimeConstantInput("scales")
                    .CompileTimeConstantInput("zero_points"),
                MlirXlaOpKernel);
REGISTER_MACHINA_MACHINA_XLA_OP(Name("UniformRequantize")
                    .CompileTimeConstantInput("input_scales")
                    .CompileTimeConstantInput("input_zero_points")
                    .CompileTimeConstantInput("output_scales")
                    .CompileTimeConstantInput("output_zero_points"),
                MlirXlaOpKernel);
REGISTER_MACHINA_MACHINA_XLA_OP(Name("UniformQuantizedAdd")
                    .CompileTimeConstantInput("lhs_scales")
                    .CompileTimeConstantInput("lhs_zero_points")
                    .CompileTimeConstantInput("rhs_scales")
                    .CompileTimeConstantInput("rhs_zero_points")
                    .CompileTimeConstantInput("output_scales")
                    .CompileTimeConstantInput("output_zero_points"),
                MlirXlaOpKernel);
REGISTER_MACHINA_MACHINA_XLA_OP(Name("UniformQuantizedClipByValue")
                    .CompileTimeConstantInput("scales")
                    .CompileTimeConstantInput("zero_points"),
                MlirXlaOpKernel);
REGISTER_MACHINA_MACHINA_XLA_OP(Name("UniformQuantizedConvolution")
                    .CompileTimeConstantInput("lhs_scales")
                    .CompileTimeConstantInput("lhs_zero_points")
                    .CompileTimeConstantInput("rhs_scales")
                    .CompileTimeConstantInput("rhs_zero_points")
                    .CompileTimeConstantInput("output_scales")
                    .CompileTimeConstantInput("output_zero_points"),
                MlirXlaOpKernel);
REGISTER_MACHINA_MACHINA_XLA_OP(Name("UniformQuantizedDot")
                    .CompileTimeConstantInput("lhs_scales")
                    .CompileTimeConstantInput("lhs_zero_points")
                    .CompileTimeConstantInput("rhs_scales")
                    .CompileTimeConstantInput("rhs_zero_points")
                    .CompileTimeConstantInput("output_scales")
                    .CompileTimeConstantInput("output_zero_points"),
                MlirXlaOpKernel);

}  // namespace
}  // namespace machina
