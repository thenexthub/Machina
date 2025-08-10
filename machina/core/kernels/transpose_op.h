/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 26, 2025.
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

#ifndef MACHINA_CORE_KERNELS_TRANSPOSE_OP_H_
#define MACHINA_CORE_KERNELS_TRANSPOSE_OP_H_

#include "machina/core/framework/op_kernel.h"
#include "machina/core/framework/tensor.h"

namespace machina {

class TransposeOp : public OpKernel {
 public:
  explicit TransposeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override;

 protected:
  virtual absl::Status DoTranspose(OpKernelContext* ctx, const Tensor& in,
                                   absl::Span<const int32> perm,
                                   Tensor* out) = 0;
  virtual bool IsConjugate() const { return false; }
};

class TransposeCpuOp : public TransposeOp {
 public:
  explicit TransposeCpuOp(OpKernelConstruction* ctx) : TransposeOp(ctx) {}

 protected:
  absl::Status DoTranspose(OpKernelContext* ctx, const Tensor& in,
                           absl::Span<const int32> perm, Tensor* out) override;
};

#if defined(INTEL_MKL)
class MklTransposeCpuOp : public TransposeOp {
 public:
  explicit MklTransposeCpuOp(OpKernelConstruction* ctx) : TransposeOp(ctx) {}

 protected:
  Status DoTranspose(OpKernelContext* ctx, const Tensor& in,
                     gtl::ArraySlice<int32> perm, Tensor* out) override;
};
#endif  // INTEL_MKL

class TransposeGpuOp : public TransposeOp {
 public:
  explicit TransposeGpuOp(OpKernelConstruction* ctx) : TransposeOp(ctx) {}

 protected:
  absl::Status DoTranspose(OpKernelContext* ctx, const Tensor& in,
                           absl::Span<const int32> perm, Tensor* out) override;
};


// Conjugating transpose ops.
class ConjugateTransposeCpuOp : public TransposeOp {
 public:
  explicit ConjugateTransposeCpuOp(OpKernelConstruction* ctx)
      : TransposeOp(ctx) {}

 protected:
  absl::Status DoTranspose(OpKernelContext* ctx, const Tensor& in,
                           absl::Span<const int32> perm, Tensor* out) override;
  bool IsConjugate() const override { return true; }
};

#if defined(INTEL_MKL)
class MklConjugateTransposeCpuOp : public TransposeOp {
 public:
  explicit MklConjugateTransposeCpuOp(OpKernelConstruction* ctx)
      : TransposeOp(ctx) {}

 protected:
  Status DoTranspose(OpKernelContext* ctx, const Tensor& in,
                     gtl::ArraySlice<int32> perm, Tensor* out) override;
  bool IsConjugate() const override { return true; }
};
#endif  // INTEL_MKL

class ConjugateTransposeGpuOp : public TransposeOp {
 public:
  explicit ConjugateTransposeGpuOp(OpKernelConstruction* ctx)
      : TransposeOp(ctx) {}

 protected:
  absl::Status DoTranspose(OpKernelContext* ctx, const Tensor& in,
                           absl::Span<const int32> perm, Tensor* out) override;
  bool IsConjugate() const override { return true; }
};


}  // namespace machina

#endif  // MACHINA_CORE_KERNELS_TRANSPOSE_OP_H_
