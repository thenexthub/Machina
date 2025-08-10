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
#include "machina/core/runtime_fallback/util/tensor_util.h"

#include <cstddef>
#include <cstdint>

#include "machina/c/tf_datatype.h"
#include "machina/c/tf_tensor.h"
#include "machina/c/tf_tensor_internal.h"
#include "machina/core/framework/tensor.h"
#include "machina/core/framework/tensor_shape.h"
#include "machina/core/framework/types.pb.h"
#include "machina/core/platform/tstring.h"
#include "machina/core/runtime_fallback/util/type_util.h"
#include "tfrt/dtype/dtype.h"  // from @tf_runtime
#include "tfrt/host_context/host_buffer.h"  // from @tf_runtime
#include "tfrt/support/forward_decls.h"  // from @tf_runtime
#include "tfrt/tensor/string_host_tensor.h"  // from @tf_runtime
#include "tfrt/tensor/tensor_shape.h"  // from @tf_runtime

namespace machina {
namespace tfd {

using tfrt::DType;
using tfrt::Expected;
using tfrt::HostBuffer;
using tfrt::RCReference;
using tfrt::StringHostTensor;

// Moves one ref on HostBuffer to machina::Tensor.
machina::Tensor MoveHostBufferToTfTensor(RCReference<HostBuffer> host_buffer,
                                            DType dtype,
                                            const tfrt::TensorShape& shape) {
  toolchain::SmallVector<tfrt::Index, 4> dims;
  shape.GetDimensions(&dims);

  auto deallocator = [](void* data, size_t len, void* arg) {
    auto* host_buffer = reinterpret_cast<HostBuffer*>(arg);
    host_buffer->DropRef();
  };

  CheckBoolCompatibility();
  // Transfer one HostBuffer ref to TFTensor.
  auto* data = host_buffer->data();
  auto size = host_buffer->size();
  OwnedTFTensor tf_tensor{TF_NewTensor(
      static_cast<TF_DataType>(GetTfDataType(dtype)), dims.data(), dims.size(),
      data, size, deallocator, host_buffer.release())};
  return TensorFromInterface(tf_tensor->tensor);
}

machina::Tensor CopyShtToTfTensor(const StringHostTensor& sht) {
  toolchain::SmallVector<tfrt::Index, 4> dims;
  sht.shape().GetDimensions(&dims);

  machina::Tensor tensor(
      machina::DT_STRING,
      machina::TensorShape(
          toolchain::SmallVector<int64_t, 4>(dims.begin(), dims.end())));

  auto len = tensor.NumElements();
  auto from = sht.strings();
  auto to = tensor.flat<machina::tstring>();

  // TODO(tfrt-devs): Consider a more efficient way to pass string
  // tensors between TFRT and TF.
  for (int i = 0; i < len; ++i) {
    to(i).assign(from[i].data(), from[i].size());
  }
  return tensor;
}

}  // namespace tfd
}  // namespace machina
