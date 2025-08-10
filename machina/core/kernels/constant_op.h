/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, March 6, 2025.
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

#ifndef MACHINA_CORE_KERNELS_CONSTANT_OP_H_
#define MACHINA_CORE_KERNELS_CONSTANT_OP_H_

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "machina/core/framework/op_kernel.h"
#include "machina/core/framework/tensor_types.h"
#include "machina/core/platform/macros.h"

namespace machina {

// ConstantOp returns a tensor specified by ConstantOpDef.
class ConstantOp : public OpKernel {
 public:
  explicit ConstantOp(OpKernelConstruction* ctx);
  void Compute(OpKernelContext* ctx) override;
  bool IsExpensive() override { return false; }
  const Tensor* const_tensor() const override { return &tensor_; };
  ~ConstantOp() override;

 private:
  Tensor tensor_;
  ConstantOp(const ConstantOp&) = delete;
  void operator=(const ConstantOp&) = delete;
};

class PlaceholderOp : public OpKernel {
 public:
  explicit PlaceholderOp(OpKernelConstruction* ctx);
  void Compute(OpKernelContext* ctx) override;

 private:
  PartialTensorShape expected_shape_;
};

}  // namespace machina

#endif  // MACHINA_CORE_KERNELS_CONSTANT_OP_H_
