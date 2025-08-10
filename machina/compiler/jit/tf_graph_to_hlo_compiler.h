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

#ifndef MACHINA_COMPILER_JIT_TF_GRAPH_TO_HLO_COMPILER_H_
#define MACHINA_COMPILER_JIT_TF_GRAPH_TO_HLO_COMPILER_H_

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "machina/compiler/jit/tf_to_hlo_compiler.h"
#include "machina/compiler/tf2xla/xla_argument.h"
#include "machina/compiler/tf2xla/xla_compiler.h"
#include "machina/compiler/tf2xla/xla_helpers.h"
#include "machina/core/framework/op_kernel.h"

namespace machina {

class TfGraphToHloCompiler : public TfToHloCompiler {
 public:
  TfGraphToHloCompiler() = delete;

  explicit TfGraphToHloCompiler(const XlaCompiler::Options& options)
      : xla_compiler_(options) {}

  // Compiles a Tensorflow `function` into an HloModuleProto stored in the
  // XlaCompilationResult pointed to by `result` by calling
  // XlaCompiler::CompileFunction.
  absl::Status Compile(const XlaCompiler::CompileOptions& options,
                       const NameAttrList& function,
                       absl::Span<const XlaArgument> args,
                       XlaCompilationResult* result) override;

  // Compiles a Tensorflow single op into an HloModuleProto stored in the
  // XlaCompilationResult pointed to by `result` by calling
  // XlaCompiler::CompileSingleOp.
  absl::Status CompileSingleOp(const XlaCompiler::CompileOptions& options,
                               const OpKernelContext* ctx,
                               absl::Span<const XlaArgument> args,
                               XlaCompilationResult* result) override;

 private:
  XlaCompiler xla_compiler_;

  TfGraphToHloCompiler(const TfGraphToHloCompiler&) = delete;
  void operator=(const TfGraphToHloCompiler&) = delete;
};

}  // namespace machina

#endif  // MACHINA_COMPILER_JIT_TF_GRAPH_TO_HLO_COMPILER_H_
