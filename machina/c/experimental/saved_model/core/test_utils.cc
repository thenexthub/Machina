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

#include "machina/c/experimental/saved_model/core/test_utils.h"

#include <memory>
#include <vector>

#include "absl/types/span.h"
#include "machina/c/eager/immediate_execution_tensor_handle.h"
#include "machina/c/tensor_interface.h"
#include "machina/core/common_runtime/device_mgr.h"
#include "machina/core/common_runtime/eager/context.h"
#include "machina/core/framework/numeric_types.h"
#include "machina/core/framework/register_types.h"
#include "machina/core/framework/tensor.h"
#include "machina/core/framework/types.h"
#include "machina/core/framework/types.pb.h"
#include "machina/core/platform/bfloat16.h"
#include "machina/core/platform/logging.h"
#include "machina/core/platform/test.h"
#include "machina/core/platform/types.h"

namespace machina {
namespace testing {

std::unique_ptr<StaticDeviceMgr> CreateTestingDeviceMgr() {
  return std::make_unique<StaticDeviceMgr>(
      DeviceFactory::NewDevice("CPU", {}, "/job:localhost/replica:0/task:0"));
}

EagerContextPtr CreateTestingEagerContext(DeviceMgr* device_mgr) {
  return EagerContextPtr(new EagerContext(
      SessionOptions(),
      machina::ContextDevicePlacementPolicy::DEVICE_PLACEMENT_SILENT,
      /* async= */ false, device_mgr,
      /* device_mgr_owned= */ false, /* rendezvous= */ nullptr,
      /* cluster_flr= */ nullptr));
}

std::vector<DataType> DataTypeSetToVector(DataTypeSet set) {
  std::vector<DataType> result;
  result.reserve(set.size());
  for (DataType dt : set) {
    result.push_back(dt);
  }
  return result;
}

std::vector<std::vector<int64_t>> InterestingShapes() {
  std::vector<std::vector<int64_t>> interesting_shapes;
  interesting_shapes.push_back({});             // Scalar
  interesting_shapes.push_back({10});           // 1D Vector
  interesting_shapes.push_back({3, 3});         // 2D Matrix
  interesting_shapes.push_back({1, 4, 6, 10});  // Higher Dimension Tensor
  return interesting_shapes;
}

ImmediateTensorHandlePtr CreateTensorHandle(ImmediateExecutionContext* ctx,
                                            DataType dtype,
                                            absl::Span<const int64_t> shape,
                                            int8_t value) {
  AbstractTensorPtr tensor(ctx->CreateTensor(dtype, shape));
  CHECK_NE(tensor.get(), nullptr)
      << "Tensor creation failed for tensor of dtype: "
      << DataTypeString(dtype);
  CHECK_EQ(tensor->Type(), dtype);
  for (int i = 0; i < shape.size(); ++i) {
    CHECK_EQ(tensor->Dim(i), shape[i]);
  }
  FillNumericTensorBuffer(tensor->Type(), tensor->NumElements(), tensor->Data(),
                          value);
  ImmediateTensorHandlePtr handle(ctx->CreateLocalHandle(tensor.get()));
  CHECK_NE(handle.get(), nullptr);
  return handle;
}

void FillNumericTensorBuffer(DataType dtype, size_t num_elements, void* buffer,
                             int8_t value) {
  switch (dtype) {
#define CASE(type)                                   \
  case DataTypeToEnum<type>::value: {                \
    type* typed_buffer = static_cast<type*>(buffer); \
    for (size_t i = 0; i < num_elements; ++i) {      \
      typed_buffer[i] = static_cast<type>(value);    \
    }                                                \
    break;                                           \
  }
    TF_CALL_INTEGRAL_TYPES(CASE);
    TF_CALL_double(CASE);
    TF_CALL_float(CASE);
    TF_CALL_int4(CASE);
    TF_CALL_uint4(CASE);
    TF_CALL_int2(CASE);
    TF_CALL_uint2(CASE);
#undef CASE
    default:
      CHECK(false) << "Unsupported data type: " << DataTypeString(dtype);
      break;
  }
}

// Checks the underlying data is equal for the buffers for two numeric tensors.
// Note: The caller must ensure to check that the dtypes and sizes of the
// underlying buffers are the same before calling this.
void CheckBufferDataIsEqual(DataType dtype, int64_t num_elements, void* a,
                            void* b) {
  switch (dtype) {
#define CASE(type)                                       \
  case DataTypeToEnum<type>::value: {                    \
    type* typed_a = static_cast<type*>(a);               \
    type* typed_b = static_cast<type*>(b);               \
    for (int64_t i = 0; i < num_elements; ++i) {         \
      if (DataTypeIsFloating(dtype)) {                   \
        EXPECT_FLOAT_EQ(static_cast<float>(typed_a[i]),  \
                        static_cast<float>(typed_b[i])); \
      } else {                                           \
        EXPECT_EQ(typed_a[i], typed_b[i]);               \
      }                                                  \
    }                                                    \
    break;                                               \
  }
    TF_CALL_INTEGRAL_TYPES(CASE);
    TF_CALL_double(CASE);
    TF_CALL_float(CASE);
    TF_CALL_int4(CASE);
    TF_CALL_uint4(CASE);
    TF_CALL_int2(CASE);
    TF_CALL_uint2(CASE);
#undef CASE
    default:
      CHECK(false) << "Unsupported data type: " << DataTypeString(dtype);
  }
}

AbstractTensorPtr TensorHandleToTensor(ImmediateExecutionTensorHandle* handle) {
  absl::Status status;
  AbstractTensorPtr tensor(handle->Resolve(&status));
  CHECK(status.ok()) << status.message();
  CHECK_NE(tensor.get(), nullptr);
  return tensor;
}

}  // namespace testing
}  // namespace machina
