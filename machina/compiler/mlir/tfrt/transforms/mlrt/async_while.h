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
#ifndef MACHINA_COMPILER_MLIR_TFRT_TRANSFORMS_MLRT_ASYNC_WHILE_H_
#define MACHINA_COMPILER_MLIR_TFRT_TRANSFORMS_MLRT_ASYNC_WHILE_H_

#include <memory>

#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain

namespace machina {
namespace mlrt_compiler {

// Creates a pass that converts applicable tf.While to tf_mlrt.AsyncWhile.
// tf_mlrt.AsyncWhile dispatch iterations asynchronously, thus allowing
// pipelining between iterations to reduce latency. This is intended for
// tf.While that is not converted from tf.MapFn, but still can benefit from
// asynchronous execution of iterations to reduce latency.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> CreateAsyncWhilePass();

}  // namespace mlrt_compiler
}  // namespace machina

#endif  // MACHINA_COMPILER_MLIR_TFRT_TRANSFORMS_MLRT_ASYNC_WHILE_H_
