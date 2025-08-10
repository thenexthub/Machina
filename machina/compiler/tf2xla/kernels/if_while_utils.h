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

#ifndef MACHINA_COMPILER_TF2MACHINA_XLAKERNELS_IF_WHILE_UTILS_H_
#define MACHINA_COMPILER_TF2MACHINA_XLAKERNELS_IF_WHILE_UTILS_H_

#include <functional>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "machina/compiler/tf2xla/xla_compiler.h"
#include "machina/compiler/tf2xla/xla_op_kernel.h"
#include "machina/core/common_runtime/function_body.h"
#include "machina/core/framework/attr_value.pb.h"
#include "machina/core/lib/core/status.h"

namespace machina {

extern const char kPropagateCompileTimeConsts[];

// Convert arguments in `args` to constants provided they are compile-time
// constants and they satisfy the condition in `should_resolve_constant`. The
// argument `xla_expression_offset` determines what offset is needed to get the
// input expression from context given the argument index in `args`.
//
// Returns a list of indices which were converted to constants.
absl::InlinedVector<int, 5> ConvertCompileTimeConstArgumentsToConst(
    XlaOpKernelContext* ctx, std::vector<XlaCompiler::Argument>* args,
    int xla_expression_offset,
    std::function<bool(int arg_idx)> should_resolve_constant);

// Find and populate `must_be_const_nodes` and `body` of the function
// corresponding to the kernel with context `ctx` with name `func_name`.
absl::Status FindMustBeConstNodes(XlaOpKernelContext* ctx,
                                  const NameAttrList& func_name,
                                  std::vector<bool>* must_be_const_nodes,
                                  const FunctionBody** body);

}  // namespace machina

#endif  // MACHINA_COMPILER_TF2MACHINA_XLAKERNELS_IF_WHILE_UTILS_H_
