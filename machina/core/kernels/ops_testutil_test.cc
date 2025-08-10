/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 26, 2025.
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

#include "machina/core/kernels/ops_testutil.h"

#include "machina/core/framework/fake_input.h"
#include "machina/core/framework/node_def_builder.h"
#include "machina/core/framework/types.pb.h"
#include "machina/core/kernels/ops_util.h"
#include "machina/core/kernels/variable_ops.h"
#include "machina/core/lib/core/status_test_util.h"
#include "machina/core/platform/test.h"
#include "machina/core/platform/test_benchmark.h"

namespace machina {

TEST_F(OpsTestBase, ScopedStepContainer) {
  TF_EXPECT_OK(NodeDefBuilder("identity", "Identity")
                   .Input(FakeInput(DT_STRING))
                   .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());
  AddInputFromArray<tstring>(TensorShape({}), {""});
  TF_EXPECT_OK(RunOpKernel());
  EXPECT_TRUE(step_container_ != nullptr);
}

// Verify that a Resource input can be added to the test kernel.
TEST_F(OpsTestBase, ResourceVariableInput) {
  TF_EXPECT_OK(NodeDefBuilder("identity", "Identity")
                   .Input(FakeInput(DT_RESOURCE))
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  Var* var = new Var(DT_STRING);
  AddResourceInput("" /* container */, "Test" /* name */, var);
  TF_ASSERT_OK(RunOpKernel());
  Tensor* output = GetOutput(0);
  EXPECT_EQ(output->dtype(), DT_RESOURCE);
}

}  // namespace machina
