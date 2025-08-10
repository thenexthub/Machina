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
#include "machina/core/tfrt/utils/tensor_util.h"

#include <assert.h>
#include <sys/types.h>

#include <cstring>
#include <string>
#include <utility>

#include "absl/container/inlined_vector.h"
#include "absl/strings/str_cat.h"
#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "machina/core/framework/tensor.h"
#include "machina/core/framework/tensor_shape.h"
#include "machina/core/framework/types.h"
#include "machina/core/framework/types.pb.h"
#include "machina/core/platform/errors.h"
#include "machina/core/platform/tstring.h"
#include "machina/core/runtime_fallback/kernel/kernel_fallback_tensor.h"
#include "machina/core/runtime_fallback/util/tensor_util.h"
#include "machina/core/runtime_fallback/util/type_util.h"
#include "tfrt/core_runtime/tensor_handle.h"  // from @tf_runtime
#include "tfrt/dtype/dtype.h"  // from @tf_runtime
#include "tfrt/host_context/host_buffer.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime
#include "tfrt/support/error_util.h"  // from @tf_runtime
#include "tfrt/support/forward_decls.h"  // from @tf_runtime
#include "tfrt/tensor/dense_host_tensor.h"  // from @tf_runtime
#include "tfrt/tensor/host_tensor.h"  // from @tf_runtime
#include "tfrt/tensor/scalar_host_tensor.h"  // from @tf_runtime
#include "tfrt/tensor/string_host_tensor.h"  // from @tf_runtime
#include "tfrt/tensor/tensor.h"  // from @tf_runtime
#include "tfrt/tensor/tensor_shape.h"  // from @tf_runtime

namespace tfrt {
namespace {

using ::machina::StatusOr;

toolchain::Expected<machina::Tensor> CopyScalarHostTensorToTFTensor(
    const AnyScalarHostTensor& tensor) {
  auto element_byte_size = GetHostSize(tensor.dtype());
  if (element_byte_size == 0) {
    return MakeStringError(
        "Failed to convert ScalarHostTensor to machina::Tensor: "
        "unsupported dtype: ",
        tensor.dtype());
  }

  toolchain::SmallVector<Index, 4> dims;
  tensor.shape().GetDimensions(&dims);

  auto tf_dtype = machina::tfd::GetTfDataType(tensor.dtype());
  machina::Tensor tf_tensor(
      tf_dtype, machina::TensorShape(
                    toolchain::SmallVector<int64_t, 4>(dims.begin(), dims.end())));

  // This can be a DCHECK instead of returninng an error because TFRT's
  // ScalarHostTensor only supports these types.
  DCHECK(DataTypeCanUseMemcpy(tf_dtype));

  // TODO(tfrt-devs): Hide the following the logic of generating a full size
  // buffer for the scalar host tensor under Tensor (and optimize if
  // necessary), so we don't have to re-implement it every time we need it.
  char* begin = reinterpret_cast<char*>(tf_tensor.data());
  for (int i = 0; i < tf_tensor.NumElements(); ++i) {
    std::memcpy(begin, tensor.data(), element_byte_size);
    begin += element_byte_size;
  }
  return tf_tensor;
}

absl::StatusOr<DType> ConvertTFDTypeToTFRTDType(machina::DataType dtype) {
  switch (dtype) {
#define DTYPE(TFRT_DTYPE, TF_DTYPE) \
  case machina::TF_DTYPE:        \
    return DType(DType::TFRT_DTYPE);
#include "machina/core/tfrt/utils/dtype.def"
    default:
      return machina::errors::Internal(absl::StrCat(
          "unsupported machina dtype: ", machina::DataType_Name(dtype)));
  }
}

absl::StatusOr<machina::DataType> ConvertTFRTDTypeToTFDType(DType dtype) {
  switch (dtype) {
#define DTYPE(TFRT_DTYPE, TF_DTYPE) \
  case DType::TFRT_DTYPE:           \
    return machina::TF_DTYPE;
#include "machina/core/tfrt/utils/dtype.def"
    default:
      return machina::errors::Internal(
          StrCat("unsupported tfrt dtype: ", dtype));
  }
}

}  // namespace

toolchain::Expected<machina::Tensor> TFRTTensorToTFTensor(const Tensor& tensor) {
  if (auto* knfbt = toolchain::dyn_cast<machina::KernelFallbackTensor>(&tensor)) {
    return *knfbt->GetTensor();
  }
  // TODO(tfrt-devs): The following logic should be better provided by
  // Tensor so we don't have to re-implement it.
  if (auto* dht = toolchain::dyn_cast<DenseHostTensor>(&tensor)) {
    return machina::tfd::MoveHostBufferToTfTensor(
        dht->buffer(), dht->dtype(), dht->shape());
  }
  if (auto* sht = toolchain::dyn_cast<StringHostTensor>(&tensor)) {
    return machina::tfd::CopyShtToTfTensor(*sht);
  }
  if (auto* scalar = toolchain::dyn_cast<AnyScalarHostTensor>(&tensor)) {
    return CopyScalarHostTensorToTFTensor(*scalar);
  }
  return MakeStringError("Unsupported conversion format for ",
                         tensor.tensor_type().name());
}

AsyncValueRef<TensorHandle> TFTensorToTFRTTensorHandle(
    const machina::Tensor& tf_tensor, HostContext* host_ctx) {
  auto knfbt =
      MakeAvailableAsyncValueRef<machina::KernelFallbackTensor>(tf_tensor);
  return MakeAvailableAsyncValueRef<TensorHandle>(
      host_ctx->GetHostDeviceRef(), knfbt->metadata(), std::move(knfbt));
}

absl::StatusOr<TensorHandle> CreateTensorHandleFromTFTensor(
    const machina::Tensor& tensor, HostContext* host) {
  // TODO(chky): Handle non-trivial types such as strings.
  TF_ASSIGN_OR_RETURN(auto dtype, ConvertTFDTypeToTFRTDType(tensor.dtype()));
  auto shape = tensor.shape().dim_sizes();
  TensorMetadata metadata(dtype, TensorShape(toolchain::SmallVector<Index, 4>(
                                     shape.begin(), shape.end())));

  if (dtype == DType::String) {
    auto sht_ref =
        StringHostTensor::MakeConstructedAsyncValueRef(metadata, host);
    auto to = sht_ref->strings();
    auto from = tensor.flat<machina::tstring>();
    for (int i = 0, e = to.size(); i < e; ++i) {
      to[i] = from(i);
    }
    sht_ref.SetStateConcrete();
    return TensorHandle(host->GetHostDeviceRef(), metadata, std::move(sht_ref));
  }

  auto dht_ref = DenseHostTensor::MakeConstructedAsyncValueRef(metadata, host);

  auto& dht = dht_ref.get();
  assert(dht.DataSizeInBytes() ==
         tensor.NumElements() * machina::DataTypeSize(tensor.dtype()));
  std::memcpy(dht_ref.get().data(), tensor.data(), dht.DataSizeInBytes());

  dht_ref.SetStateConcrete();
  return TensorHandle(host->GetHostDeviceRef(), metadata, std::move(dht_ref));
}

absl::StatusOr<machina::Tensor> CreateTFTensorFromTensorHandle(
    const TensorHandle& tensor_handle) {
  const auto& metadata = tensor_handle.GetAvailableMetadata();
  TF_ASSIGN_OR_RETURN(auto dtype, ConvertTFRTDTypeToTFDType(metadata.dtype));
  toolchain::SmallVector<Index, 4> shape;
  metadata.shape.GetDimensions(&shape);
  const auto& host_tensor = tensor_handle.GetAsyncTensor()->get<HostTensor>();

  if (auto* kernel_fallback_tensor =
          toolchain::dyn_cast<machina::KernelFallbackTensor>(&host_tensor)) {
    return *kernel_fallback_tensor->GetTensor();
  }

  if (toolchain::isa<StringHostTensor>(host_tensor)) {
    assert(dtype == machina::DT_STRING);
    const auto& sht = toolchain::cast<StringHostTensor>(host_tensor);
    machina::Tensor tensor(
        machina::DT_STRING,
        machina::TensorShape(
            toolchain::SmallVector<int64_t, 4>(shape.begin(), shape.end())));
    auto from = sht.strings();
    auto to = tensor.flat<machina::tstring>();
    for (int i = 0, e = from.size(); i < e; ++i) {
      to(i).assign(from[i].data(), from[i].size());
    }
    return tensor;
  }

  if (toolchain::isa<DenseHostTensor>(host_tensor)) {
    const auto& dht = toolchain::cast<DenseHostTensor>(host_tensor);
    machina::Tensor tensor(
        dtype, machina::TensorShape(
                   toolchain::SmallVector<int64_t, 4>(shape.begin(), shape.end())));

    assert(dht.DataSizeInBytes() ==
           tensor.NumElements() * machina::DataTypeSize(tensor.dtype()));
    std::memcpy(tensor.data(), dht.data(), dht.DataSizeInBytes());
    return tensor;
  }

  return machina::errors::Internal("unknown host tensor type");
}

Expected<tfrt::DenseHostTensor> ConvertTfTensorToDHT(
    machina::Tensor tf_tensor) {
  auto metadata = machina::tfd::GetTensorMetadata(tf_tensor);
  if (!IsTriviallyCopyable(metadata.dtype))
    return MakeStringError(
        "Cannot convert tf Tensor with non-trivially copyable dtype to DHT");

  void* data = tf_tensor.data();
  size_t size = tf_tensor.AllocatedBytes();
  tfrt::RCReference<tfrt::HostBuffer> host_buffer =
      tfrt::HostBuffer::CreateFromExternal(
          data, size, [tf_tensor = std::move(tf_tensor)](void*, size_t) {});

  // Assume HostBuffer::CreateFromExternal never fails.
  assert(host_buffer);

  return tfrt::DenseHostTensor(metadata, std::move(host_buffer));
}

}  // namespace tfrt
