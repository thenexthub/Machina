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

#include <cstdint>
#include <memory>
#include <tuple>
#include <vector>

#include "absl/status/status.h"
#include "machina/c/eager/immediate_execution_tensor_handle.h"
#include "machina/c/experimental/saved_model/core/revived_types/constant.h"
#include "machina/c/experimental/saved_model/core/saved_model_utils.h"
#include "machina/c/experimental/saved_model/core/test_utils.h"
#include "machina/core/common_runtime/device_mgr.h"
#include "machina/core/common_runtime/eager/context.h"
#include "machina/core/framework/tensor.h"
#include "machina/core/framework/tensor.pb.h"
#include "machina/core/framework/tensor_shape.h"
#include "machina/core/framework/types.h"
#include "machina/core/framework/types.pb.h"
#include "machina/core/lib/core/status_test_util.h"
#include "machina/core/platform/test.h"

namespace machina {
namespace {

class ConstantTest : public ::testing::TestWithParam<
                         std::tuple<DataType, std::vector<int64_t>, bool>> {
 public:
  ConstantTest()
      : device_mgr_(testing::CreateTestingDeviceMgr()),
        ctx_(testing::CreateTestingEagerContext(device_mgr_.get())) {}

  EagerContext* context() { return ctx_.get(); }

 private:
  std::unique_ptr<StaticDeviceMgr> device_mgr_;
  EagerContextPtr ctx_;
};

// Basic sanity check that roundtripping a Tensor->Tensorproto->Constant
// preserves values.
TEST_P(ConstantTest, CreateConstantSuccessful) {
  // Get test parameters
  auto& test_params = GetParam();
  DataType dtype = std::get<0>(test_params);
  TensorShape shape(std::get<1>(test_params));
  bool tensorproto_use_tensor_content = std::get<2>(test_params);

  // Construct a Tensor with the given dtype + shape
  Tensor expected(dtype, shape);
  testing::FillNumericTensorBuffer(expected.dtype(), expected.NumElements(),
                                   expected.data(), 42);

  // Serialize it to a Tensorproto
  TensorProto proto;
  if (tensorproto_use_tensor_content) {
    expected.AsProtoTensorContent(&proto);
  } else {
    expected.AsProtoField(&proto);
  }

  // Revival should succeed w/o errors
  std::unique_ptr<Constant> revived;
  TF_EXPECT_OK(internal::TensorProtoToConstant(context(), proto, &revived));

  // The revived tensorhandle should have the exact same dtype, shape, +
  // approx equivalent data to the original.
  ImmediateExecutionTensorHandle* handle = revived->handle();
  absl::Status status;
  AbstractTensorPtr revived_tensor(handle->Resolve(&status));
  TF_EXPECT_OK(status) << "Failed to convert tensorhandle to tensor";
  EXPECT_EQ(revived_tensor->Type(), expected.dtype());
  EXPECT_EQ(revived_tensor->NumElements(), expected.NumElements());
  EXPECT_EQ(revived_tensor->NumDims(), expected.dims());
  for (int i = 0; i < expected.dims(); ++i) {
    EXPECT_EQ(revived_tensor->Dim(i), expected.dim_size(i));
  }

  testing::CheckBufferDataIsEqual(expected.dtype(), expected.NumElements(),
                                  revived_tensor->Data(), expected.data());
}

// Test against combinations of tensors that are
// 1. Varying dtypes
// 2. Varying shapes
// 3. TensorProto serialized using tensor_content vs repeated type
INSTANTIATE_TEST_SUITE_P(
    ConstantIntegerDtypesTest, ConstantTest,
    ::testing::Combine(
        ::testing::ValuesIn(testing::DataTypeSetToVector(kDataTypeIsInteger)),
        ::testing::ValuesIn(testing::InterestingShapes()),
        ::testing::Values(false, true)));

INSTANTIATE_TEST_SUITE_P(
    ConstantFloatingDtypesTest, ConstantTest,
    ::testing::Combine(::testing::Values(DT_FLOAT, DT_DOUBLE),
                       ::testing::ValuesIn(testing::InterestingShapes()),
                       ::testing::Values(false, true)));

}  // namespace
}  // namespace machina
