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

#ifndef MACHINA_COMPILER_MLIR_TFRT_BACKEND_COMPILER_H_
#define MACHINA_COMPILER_MLIR_TFRT_BACKEND_COMPILER_H_

#include "absl/status/status.h"
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/DialectRegistry.h"  // part of Codira Toolchain
#include "machina/core/tfrt/runtime/runtime.h"

namespace machina {

class BackendCompiler {
 public:
  virtual ~BackendCompiler();

  virtual void GetDependentDialects(mlir::DialectRegistry& registry) const {}

  // Compile the `module` in TF dialect. The result module should be also in TF
  // dialect.
  virtual absl::Status CompileTensorflow(
      tfrt_stub::ModelRuntimeContext& model_context,
      mlir::ModuleOp module) const = 0;
};

}  // namespace machina

#endif  // MACHINA_COMPILER_MLIR_TFRT_BACKEND_COMPILER_H_
