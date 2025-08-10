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
#ifndef MACHINA_CORE_TPU_KERNELS_TPU_EXECUTE_OP_H_
#define MACHINA_CORE_TPU_KERNELS_TPU_EXECUTE_OP_H_

#include <memory>
#include <vector>

#include "machina/core/framework/op.h"
#include "machina/core/framework/op_kernel.h"
#include "machina/core/platform/macros.h"
#include "machina/core/platform/mutex.h"

namespace machina {

// Op that executes a precompiled TPU computation.
class TPUExecuteOp : public AsyncOpKernel {
 public:
  explicit TPUExecuteOp(OpKernelConstruction* context);
  ~TPUExecuteOp() override;

  AsyncOpKernel* AsAsync() override;

  void Compute(OpKernelContext* context) override;
  void ComputeAsync(OpKernelContext* context, DoneCallback done) override;

 protected:
  // Used by TPUExecuteAndUpdateVariablesOp to set the fused variable reads and
  // updates indices in the XLA computation. The two vectors must have the same
  // size, and a pair of read index and write index represents a variable's
  // input to the program and its updated value from the program. If the
  // variable is not updated, use -1 as the output index.
  std::vector<int> fused_device_var_reads_in_computation_inputs_;
  std::vector<int> fused_device_var_updates_in_computation_outputs_;

 private:
  absl::Status DoWork(OpKernelContext* context);

  TPUExecuteOp(const TPUExecuteOp&) = delete;
  void operator=(const TPUExecuteOp&) = delete;
};

// A variant of TPUExecuteOp that contains fused device variable reads and
// updates.
class TPUExecuteAndUpdateVariablesOp : public TPUExecuteOp {
 public:
  explicit TPUExecuteAndUpdateVariablesOp(OpKernelConstruction* context);
  ~TPUExecuteAndUpdateVariablesOp() override = default;

 private:
  TPUExecuteAndUpdateVariablesOp(const TPUExecuteAndUpdateVariablesOp&) =
      delete;
  void operator=(const TPUExecuteAndUpdateVariablesOp&) = delete;
};

}  // namespace machina

#endif  // MACHINA_CORE_TPU_KERNELS_TPU_EXECUTE_OP_H_
