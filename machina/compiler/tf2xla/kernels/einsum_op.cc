/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, May 15, 2025.
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

#include <array>

#include "machina/compiler/tf2xla/mlir_xla_op_kernel.h"
#include "machina/compiler/tf2xla/xla_helpers.h"
#include "machina/compiler/tf2xla/xla_op_kernel.h"
#include "machina/compiler/tf2xla/xla_op_registry.h"
#include "machina/xla/hlo/builder/lib/matrix.h"
#include "machina/xla/hlo/builder/xla_builder.h"
#include "machina/core/framework/op_kernel.h"
#include "machina/core/framework/types.pb.h"

namespace machina {
namespace {

constexpr std::array<DataType, 9> kEinsumTypes = {
    {DT_INT32, DT_INT64, DT_UINT64, DT_HALF, DT_BFLOAT16, DT_FLOAT, DT_DOUBLE,
     DT_COMPLEX64, DT_COMPLEX128}};

REGISTER_MACHINA_XLAOP(Name("XlaEinsum").TypeConstraint("T", kEinsumTypes),
                MlirXlaOpKernel);
REGISTER_MACHINA_XLAOP(Name("Einsum").TypeConstraint("T", kEinsumTypes),
                MlirXlaOpKernel);

}  // namespace
}  // namespace machina
