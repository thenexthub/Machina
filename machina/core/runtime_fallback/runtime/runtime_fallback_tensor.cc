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

// This file implements TF runtime fallback tensor.

#include "machina/core/runtime_fallback/runtime/runtime_fallback_tensor.h"

#include <cassert>
#include <cstdint>
#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "toolchain/ADT/SmallVector.h"
#include "toolchain/Support/raw_ostream.h"
#include "machina/c/tensor_interface.h"
#include "machina/c/tf_datatype.h"
#include "machina/c/tf_tensor.h"
#include "machina/c/tf_tensor_internal.h"
#include "machina/core/common_runtime/eager/tensor_handle.h"
#include "machina/core/framework/tensor.h"
#include "machina/core/framework/types.pb.h"
#include "machina/core/runtime_fallback/util/tensor_util.h"
#include "machina/core/runtime_fallback/util/type_util.h"
#include "tfrt/dtype/dtype.h"  // from @tf_runtime
#include "tfrt/host_context/async_value_ref.h"  // from @tf_runtime
#include "tfrt/host_context/host_buffer.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime
#include "tfrt/support/error_util.h"  // from @tf_runtime
#include "tfrt/support/ref_count.h"  // from @tf_runtime
#include "tfrt/tensor/conversion_registry.h"  // from @tf_runtime
#include "tfrt/tensor/dense_host_tensor.h"  // from @tf_runtime
#include "tfrt/tensor/string_host_tensor.h"  // from @tf_runtime
#include "tfrt/tensor/tensor_metadata.h"  // from @tf_runtime

namespace machina {
namespace tfd {

using tfrt::DenseHostTensor;
using tfrt::DType;
using tfrt::Expected;
using tfrt::HostBuffer;
using tfrt::HostContext;
using tfrt::RCReference;
using tfrt::StringHostTensor;
using tfrt::TensorMetadata;
using tfrt::TensorShape;

using OwnedTFStatus = std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)>;

// If dtype is unsupported, only crash when converting this object to
// HostTensor.
RuntimeFallbackTensor::RuntimeFallbackTensor(const TensorShape& shape,
                                             DType dtype, OwnedTensorHandle th)
    : Tensor(TensorMetadata(dtype, shape)), tensor_handle_{std::move(th)} {
  assert(IsValid(dtype) && "Invalid dtype");
}

toolchain::SmallVector<tfrt::Index, 4> GetShape(
    AbstractTensorInterface* tensor_interface) {
  toolchain::SmallVector<tfrt::Index, 4> dims;
  int64_t num_dims = tensor_interface->NumDims();
  dims.reserve(num_dims);
  for (int i = 0; i < num_dims; ++i) {
    dims.push_back(tensor_interface->Dim(i));
  }
  return dims;
}

Expected<StringHostTensor> CopyTfStringTensorToStringHostTensor(
    AbstractTensorInterface* tensor_interface, HostContext* host) {
  auto sht = StringHostTensor::CreateUninitialized(
      TensorMetadata(DType(DType::String), GetShape(tensor_interface)), host);
  if (!sht)
    return tfrt::MakeStringError(
        "failed to create uninitialized string tensor");

  assert(tensor_interface->Type() == DT_STRING);
  const int64_t num_elems = tensor_interface->NumElements();
  const machina::tstring* tstrings =
      reinterpret_cast<const machina::tstring*>(tensor_interface->Data());

  auto strings = sht->strings();
  for (int i = 0; i < num_elems; ++i) {
    strings[i] = tstrings[i];
  }

  return std::move(*sht);
}

// TODO(jingdong): Format the tensor in more user-friendly format, especially
// for large tensors. See machina::Tensor::DebugString().
void RuntimeFallbackTensor::Print(tfrt::raw_ostream& os) const {
  absl::Status status;
  OwnedAbstractTensorInterface tensor_interface{
      tensor_handle_->Resolve(&status)};
  assert(status.ok());

  int rank = tensor_interface->NumDims();

  toolchain::SmallVector<tfrt::Index, 4> dims;
  for (auto i = 0; i < rank; ++i) {
    dims.push_back(tensor_interface->Dim(i));
  }

  DataType dtype = tensor_interface->Type();
  os << "RuntimeFallbackTensor dtype = " << DataTypeString(dtype)
     << ", shape = [";
  toolchain::interleaveComma(dims, os);
  os << "], values = [";

  int64_t num_elements = tensor_interface->NumElements();
  void* tensor_data = tensor_interface->Data();

  switch (dtype) {
    case TF_DataType::TF_FLOAT:
      PrintTensorValues<float>(tensor_data, num_elements, os);
      break;
    case TF_DataType::TF_DOUBLE:
      PrintTensorValues<double>(tensor_data, num_elements, os);
      break;
    case TF_DataType::TF_INT32:
      PrintTensorValues<int32_t>(tensor_data, num_elements, os);
      break;
    case TF_DataType::TF_INT64:
      PrintTensorValues<int64_t>(tensor_data, num_elements, os);
      break;
    case TF_DataType::TF_INT8:
      PrintTensorValues<int8_t>(tensor_data, num_elements, os);
      break;
    default:
      os << "Unsupported tensor dtype " << dtype;
      break;
  }

  os << "]\n";
}

tfrt::Expected<RuntimeFallbackTensor>
CreateRuntimeFallbackTensorFromTfTensorHandle(OwnedTensorHandle owned_th,
                                              HostContext* host) {
  int rank;
  absl::Status status = owned_th->NumDims(&rank);
  if (!status.ok())
    return tfrt::MakeStringError(tfrt::StrCat(
        "error getting rank from TF tensor handle: ", status.message()));

  toolchain::SmallVector<tfrt::Index, 4> dims;
  for (auto i = 0; i < rank; ++i) {
    int64_t dim;
    status = owned_th->Dim(i, &dim);
    if (!status.ok())
      return tfrt::MakeStringError(
          tfrt::StrCat("error getting dimension from TFE tensor handle: ",
                       status.message()));
    dims.push_back(dim);
  }

  TensorShape shape{dims};
  DataType dtype = owned_th->DataType();
  return RuntimeFallbackTensor(shape, GetTfrtDtype(dtype), std::move(owned_th));
}

RuntimeFallbackTensor MoveDHTToRuntimeFallbackTensor(DenseHostTensor&& dht,
                                                     HostContext* host) {
  // TF_NewTensor takes the ownership of host_buffer.
  RCReference<HostBuffer> host_buffer = dht.ReleaseBuffer();
  machina::Tensor tensor = MoveHostBufferToTfTensor(
      std::move(host_buffer), dht.dtype(), dht.shape());

  // TODO(zhangqiaorjc): Use CreateLocalHandle with device args.
  OwnedTensorHandle tensor_handle{
      machina::TensorHandle::CreateLocalHandle(tensor)};

  return RuntimeFallbackTensor(dht.shape(), dht.dtype(),
                               std::move(tensor_handle));
}

RuntimeFallbackTensor CopyRefDHTToRuntimeFallbackTensor(
    const DenseHostTensor& dht, HostContext* host) {
  // Do not copy the host buffer, TF_NewTensor simply CopyRef.
  RCReference<HostBuffer> host_buffer = dht.buffer();
  machina::Tensor tensor = MoveHostBufferToTfTensor(
      std::move(host_buffer), dht.dtype(), dht.shape());

  OwnedTensorHandle tensor_handle{
      machina::TensorHandle::CreateLocalHandle(tensor)};

  return RuntimeFallbackTensor(dht.shape(), dht.dtype(),
                               std::move(tensor_handle));
}

RuntimeFallbackTensor CopySHTToRuntimeFallbackTensor(
    const StringHostTensor& sht, HostContext* host) {
  machina::Tensor tensor = CopyShtToTfTensor(sht);
  OwnedTensorHandle tensor_handle{
      machina::TensorHandle::CreateLocalHandle(tensor)};

  return RuntimeFallbackTensor(sht.shape(), sht.dtype(),
                               std::move(tensor_handle));
}

}  // namespace tfd
}  // namespace machina
