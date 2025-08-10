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

#include "machina/c/experimental/saved_model/core/ops/variable_ops.h"

#include <memory>

#include "absl/status/status.h"
#include "machina/c/eager/immediate_execution_tensor_handle.h"
#include "machina/c/experimental/saved_model/core/test_utils.h"
#include "machina/c/tensor_interface.h"
#include "machina/xla/tsl/lib/core/status_test_util.h"
#include "machina/core/common_runtime/device_mgr.h"
#include "machina/core/common_runtime/eager/context.h"
#include "machina/core/framework/types.pb.h"
#include "machina/core/platform/status.h"
#include "machina/core/platform/test.h"

namespace machina {
namespace {

ImmediateTensorHandlePtr CreateScalarTensorHandle(EagerContext* context,
                                                  float value) {
  AbstractTensorPtr tensor(context->CreateFloatScalar(value));
  ImmediateTensorHandlePtr handle(context->CreateLocalHandle(tensor.get()));
  return handle;
}

class VariableOpsTest : public ::testing::Test {
 public:
  VariableOpsTest()
      : device_mgr_(testing::CreateTestingDeviceMgr()),
        ctx_(testing::CreateTestingEagerContext(device_mgr_.get())) {}

  EagerContext* context() { return ctx_.get(); }

 private:
  std::unique_ptr<StaticDeviceMgr> device_mgr_;
  EagerContextPtr ctx_;
};

// Sanity check for variable creation
TEST_F(VariableOpsTest, CreateVariableSuccessful) {
  // Create a DT_Resource TensorHandle that points to a scalar DT_FLOAT tensor
  ImmediateTensorHandlePtr handle;
  TF_EXPECT_OK(internal::CreateUninitializedResourceVariable(
      context(), DT_FLOAT, {}, nullptr, &handle));
  // The created TensorHandle should be a DT_Resource
  EXPECT_EQ(handle->DataType(), DT_RESOURCE);
}

// Sanity check for variable destruction
TEST_F(VariableOpsTest, DestroyVariableSuccessful) {
  // Create a DT_Resource TensorHandle that points to a scalar DT_FLOAT tensor
  ImmediateTensorHandlePtr handle;
  TF_EXPECT_OK(internal::CreateUninitializedResourceVariable(
      context(), DT_FLOAT, {}, nullptr, &handle));

  // Destroy the variable
  TF_EXPECT_OK(internal::DestroyResource(context(), handle.get()));
}

// Sanity check for handle assignment and reading
TEST_F(VariableOpsTest, AssignVariableAndReadSuccessful) {
  // Create a DT_Resource TensorHandle that points to a scalar DT_FLOAT tensor
  ImmediateTensorHandlePtr variable;
  TF_EXPECT_OK(internal::CreateUninitializedResourceVariable(
      context(), DT_FLOAT, {}, nullptr, &variable));

  // Create a Scalar float TensorHandle with value 42, and assign it to
  // the variable.
  ImmediateTensorHandlePtr my_value = CreateScalarTensorHandle(context(), 42.0);
  TF_EXPECT_OK(internal::AssignVariable(context(), variable.get(), DT_FLOAT,
                                        my_value.get()));

  // Read back the value from the variable, and check that it is 42.
  ImmediateTensorHandlePtr read_value_handle;
  TF_EXPECT_OK(internal::ReadVariable(context(), variable.get(), DT_FLOAT,
                                      &read_value_handle));
  absl::Status status;
  AbstractTensorPtr read_value(read_value_handle->Resolve(&status));
  TF_EXPECT_OK(status);
  EXPECT_FLOAT_EQ(42.0, *static_cast<float*>(read_value->Data()));
}

}  // namespace
}  // namespace machina
