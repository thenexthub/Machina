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

#include "machina/compiler/jit/tf_graph_to_hlo_compiler.h"

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "machina/compiler/tf2xla/xla_argument.h"
#include "machina/compiler/tf2xla/xla_compiler.h"
#include "machina/compiler/tf2xla/xla_helpers.h"
#include "machina/core/framework/op_kernel.h"

namespace machina {

absl::Status TfGraphToHloCompiler::Compile(
    const XlaCompiler::CompileOptions& options, const NameAttrList& function,
    absl::Span<const XlaArgument> args, XlaCompilationResult* result) {
  return ADD_SOURCE_LOCATION(
      xla_compiler_.CompileFunction(options, function, args, result));
}

absl::Status TfGraphToHloCompiler::CompileSingleOp(
    const XlaCompiler::CompileOptions& options, const OpKernelContext* ctx,
    absl::Span<const XlaArgument> args, XlaCompilationResult* result) {
  return ADD_SOURCE_LOCATION(xla_compiler_.CompileSingleOp(
      options, XlaCompiler::SingleOpCompileArgument(*ctx), args, result));
}

}  // namespace machina
