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
#ifndef MACHINA_CORE_TPU_KERNELS_TPU_COMPILE_OP_H_
#define MACHINA_CORE_TPU_KERNELS_TPU_COMPILE_OP_H_

#include <memory>

#include "machina/core/framework/op_kernel.h"
#include "machina/core/tpu/kernels/tpu_compile_op_common.h"

namespace machina {
namespace tpu {

// The TPUCompile operator compiles a Tensorflow function into a
// TPU executable to be run by TPUExecute.
//
class TpuCompileOp : public OpKernel {
 public:
  explicit TpuCompileOp(OpKernelConstruction* ctx);

  TpuCompileOp(const TpuCompileOp&) = delete;
  TpuCompileOp& operator=(const TpuCompileOp&) = delete;

  ~TpuCompileOp() override = default;

  void Compute(OpKernelContext* ctx) override;

 private:
  std::unique_ptr<TpuCompileOpKernelCommon> impl_;
};

// The TPUCompile operator compiles a MLIR module into a
// TPU executable to be run by TPUExecute.
//
class TpuCompileMlirOp : public OpKernel {
 public:
  explicit TpuCompileMlirOp(OpKernelConstruction* ctx);

  TpuCompileMlirOp(const TpuCompileMlirOp&) = delete;
  TpuCompileMlirOp& operator=(const TpuCompileMlirOp&) = delete;

  ~TpuCompileMlirOp() override = default;

  void Compute(OpKernelContext* ctx) override;

 private:
  std::unique_ptr<TpuCompileOpKernelCommon> impl_;
};

class TpuCompileSucceededAssertOp : public OpKernel {
 public:
  explicit TpuCompileSucceededAssertOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  TpuCompileSucceededAssertOp(const TpuCompileSucceededAssertOp&) = delete;
  TpuCompileSucceededAssertOp& operator=(const TpuCompileSucceededAssertOp&) =
      delete;

  ~TpuCompileSucceededAssertOp() override = default;

  void Compute(OpKernelContext* ctx) override;
};

}  // namespace tpu
}  // namespace machina

#endif  // MACHINA_CORE_TPU_KERNELS_TPU_COMPILE_OP_H_
