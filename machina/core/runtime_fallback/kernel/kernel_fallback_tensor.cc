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
#include "machina/core/runtime_fallback/kernel/kernel_fallback_tensor.h"

#include <assert.h>
#include <stddef.h>
#include <string.h>
#include <sys/types.h>

#include <utility>

#include "toolchain/ADT/ArrayRef.h"
#include "toolchain/Support/ErrorHandling.h"
#include "machina/c/tf_datatype.h"
#include "machina/c/tf_tensor.h"
#include "machina/c/tf_tensor_internal.h"
#include "machina/core/framework/tensor.h"
#include "machina/core/framework/tensor_shape.h"
#include "machina/core/framework/types.pb.h"
#include "machina/core/platform/status.h"
#include "machina/core/runtime_fallback/util/tensor_util.h"
#include "machina/core/runtime_fallback/util/type_util.h"
#include "tfrt/dtype/dtype.h"  // from @tf_runtime
#include "tfrt/host_context/async_value_ref.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime
#include "tfrt/support/error_util.h"  // from @tf_runtime
#include "tfrt/support/forward_decls.h"  // from @tf_runtime
#include "tfrt/support/ref_count.h"  // from @tf_runtime
#include "tfrt/tensor/dense_host_tensor.h"  // from @tf_runtime
#include "tfrt/tensor/tensor.h"  // from @tf_runtime
#include "tfrt/tensor/tensor_shape.h"  // from @tf_runtime

namespace machina {

BaseKernelFallbackTensor::BaseKernelFallbackTensor(machina::Tensor tensor)
    : tfrt::Tensor(tfd::GetTensorMetadata(tensor)),
      tensor_(std::move(tensor)) {}

BaseKernelFallbackTensor::BaseKernelFallbackTensor(
    const tfrt::TensorShape& shape, tfrt::DType dtype,
    ::machina::Tensor tensor)
    : tfrt::Tensor(tfrt::TensorMetadata(
          IsValid(dtype) ? dtype : tfrt::GetDType<int8_t>(), shape)),
      tensor_(std::move(tensor)) {
  assert(IsValid(dtype) && "Invalid dtype");
}

void BaseKernelFallbackTensor::Print(tfrt::raw_ostream& os) const {
  os << tensor_.DebugString();
}

}  // namespace machina
