/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Saturday, June 21, 2025.
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

// This file declares TF kernel fallback tensor.

#ifndef MACHINA_CORE_RUNTIME_FALLBACK_KERNEL_KERNEL_FALLBACK_TENSOR_H_
#define MACHINA_CORE_RUNTIME_FALLBACK_KERNEL_KERNEL_FALLBACK_TENSOR_H_

#include <utility>

#include "machina/core/framework/tensor.h"
#include "tfrt/dtype/dtype.h"  // from @tf_runtime
#include "tfrt/support/forward_decls.h"  // from @tf_runtime
#include "tfrt/tensor/tensor.h"  // from @tf_runtime
#include "tfrt/tensor/tensor_shape.h"  // from @tf_runtime

namespace machina {

class BaseKernelFallbackTensor : public tfrt::Tensor {
 public:
  explicit BaseKernelFallbackTensor(::machina::Tensor tensor);
  BaseKernelFallbackTensor(const tfrt::TensorShape& shape, tfrt::DType dtype,
                           ::machina::Tensor tensor);

  void Print(tfrt::raw_ostream& os) const override;

  const ::machina::Tensor* GetTensor() const { return &tensor_; }

 private:
  ::machina::Tensor tensor_;
  bool is_valid_type_;
};

class KernelFallbackTensor final
    : public BaseKernelFallbackTensor,
      public tfrt::TensorTraits<KernelFallbackTensor> {
 public:
  explicit KernelFallbackTensor(::machina::Tensor tensor)
      : BaseKernelFallbackTensor(std::move(tensor)) {}
  KernelFallbackTensor(const tfrt::TensorShape& shape, tfrt::DType dtype,
                       ::machina::Tensor tensor)
      : BaseKernelFallbackTensor(shape, dtype, std::move(tensor)) {}

  static KernelFallbackTensor Create(const machina::Tensor& tensor) {
    return KernelFallbackTensor(tensor);
  }

  // Tensor type name for KernelFallbackTensor.
  static const char* name() { return "KernelFallback"; }
};

}  // namespace machina

#endif  // MACHINA_CORE_RUNTIME_FALLBACK_KERNEL_KERNEL_FALLBACK_TENSOR_H_
