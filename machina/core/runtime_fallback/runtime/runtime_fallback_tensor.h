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

// This file declares TF runtime fallback tensor.

#ifndef MACHINA_CORE_RUNTIME_FALLBACK_RUNTIME_RUNTIME_FALLBACK_TENSOR_H_
#define MACHINA_CORE_RUNTIME_FALLBACK_RUNTIME_RUNTIME_FALLBACK_TENSOR_H_

#include "toolchain/ADT/STLExtras.h"
#include "machina/core/runtime_fallback/runtime/kernel_utils.h"
#include "tfrt/support/forward_decls.h"  // from @tf_runtime
#include "tfrt/tensor/dense_host_tensor.h"  // from @tf_runtime
#include "tfrt/tensor/host_tensor.h"  // from @tf_runtime
#include "tfrt/tensor/string_host_tensor.h"  // from @tf_runtime
#include "tfrt/tensor/tensor.h"  // from @tf_runtime

namespace machina {
namespace tfd {

class RuntimeFallbackTensor final
    : public tfrt::Tensor,
      public tfrt::TensorTraits<RuntimeFallbackTensor> {
 public:
  explicit RuntimeFallbackTensor(const tfrt::TensorShape& shape,
                                 tfrt::DType dtype, OwnedTensorHandle th);

  void Print(tfrt::raw_ostream& os) const override;

  // Note that this method does not add ref to the return tensor_handle.
  TensorHandle* GetTensorHandle() const { return tensor_handle_.get(); }

  // Tensor type name for RuntimeFallbackTensor.
  static const char* name() { return "RuntimeFallback"; }

 private:
  template <typename T>
  static void PrintTensorValues(void* data, ssize_t size,
                                toolchain::raw_ostream& os) {
    toolchain::ArrayRef<T> elements = toolchain::ArrayRef(static_cast<T*>(data), size);
    toolchain::interleaveComma(elements, os);
  }

  OwnedTensorHandle tensor_handle_;
};

toolchain::SmallVector<tfrt::Index, 4> GetShape(
    AbstractTensorInterface* tensor_interface);

tfrt::Expected<tfrt::StringHostTensor> CopyTfStringTensorToStringHostTensor(
    AbstractTensorInterface* tensor_interface, tfrt::HostContext* host);

tfrt::Expected<RuntimeFallbackTensor>
CreateRuntimeFallbackTensorFromTfTensorHandle(OwnedTensorHandle owned_th,
                                              tfrt::HostContext* host);

RuntimeFallbackTensor MoveDHTToRuntimeFallbackTensor(
    tfrt::DenseHostTensor&& dht, tfrt::HostContext* host);

RuntimeFallbackTensor CopyRefDHTToRuntimeFallbackTensor(
    const tfrt::DenseHostTensor& dht, tfrt::HostContext* host);

RuntimeFallbackTensor CopySHTToRuntimeFallbackTensor(
    const tfrt::StringHostTensor& sht, tfrt::HostContext* host);

}  // namespace tfd
}  // namespace machina

#endif  // MACHINA_CORE_RUNTIME_FALLBACK_RUNTIME_RUNTIME_FALLBACK_TENSOR_H_
