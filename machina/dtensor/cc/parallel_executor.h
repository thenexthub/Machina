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

#ifndef MACHINA_DTENSOR_CC_PARALLEL_EXECUTOR_H_
#define MACHINA_DTENSOR_CC_PARALLEL_EXECUTOR_H_

#include <memory>
#include <optional>
#include <vector>

#include "toolchain/ADT/StringRef.h"
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "machina/c/eager/c_api_experimental.h"
#include "machina/xla/pjrt/pjrt_future.h"
#include "machina/dtensor/cc/tensor_layout.h"
#include "machina/dtensor/cc/tensor_with_layout.h"

namespace machina {
namespace dtensor {

template <typename T = void>
using Future = ::xla::PjRtFuture<T>;

// ParallelExecutor Interface
// Note: The interface is under development and APIs are subject to change.
class ParallelExecutor {
 public:
  virtual ~ParallelExecutor() = default;

  // Broadcasts `tensor` to `mesh` using replicated sharding and returns a
  // DTensor representation. `mesh` can be a single device mesh and in that case
  // `const_value` is useless.
  virtual StatusOr<std::unique_ptr<TensorWithLayout>> Broadcast(
      const Tensor& tensor, const Mesh& mesh,
      std::optional<NodeDef> const_value) = 0;

  // Takes input TensorWithLayouts and a MLIR module.
  // The MLIR module should have `main` as its entry function name.
  // Attributes are forwarded to executed operations unmodified.
  // The execute is non-blocking and returns a Future of output TensorWithLayout
  // raw pointers.
  // The client is responsible for the ownership of the outputs.
  struct ExecutionResult {
    Future<> status;
    // The pointed data of `outputs` are filled after `status` future resolves
    // as ok.
    std::vector<TensorWithLayout*> outputs;
  };
  virtual StatusOr<ExecutionResult> Execute(
      TFE_Context* context, const std::vector<TensorWithLayout*>& inputs,
      mlir::ModuleOp module, const TFE_OpAttrs* attributes) const = 0;

  // Disassembles `t` into multiple TensorWithLayouts. `t` may or may not be
  // valid to use afterwards.
  virtual StatusOr<std::vector<std::unique_ptr<TensorWithLayout>>> Disassemble(
      TensorWithLayout* t) = 0;

  // Returns a tensor copied from `t` when `t` contains only a single device.
  virtual Future<Tensor> ToHostBuffer(TensorWithLayout* t) = 0;
};

// Factory method for Default ParallelExecutor instance.
StatusOr<std::unique_ptr<ParallelExecutor>> CreateDefaultParallelExecutor();

}  // namespace dtensor
}  // namespace machina

#endif  // MACHINA_DTENSOR_CC_PARALLEL_EXECUTOR_H_
