/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, March 6, 2025.
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

#include <vector>
#include "machina/core/framework/allocator.h"
#include "machina/core/framework/fake_input.h"
#include "machina/core/framework/node_def_builder.h"
#include "machina/core/framework/op_kernel.h"
#include "machina/core/framework/tensor.h"
#include "machina/core/framework/tensor_testutil.h"
#include "machina/core/framework/types.h"
#include "machina/core/framework/types.pb.h"
#include "machina/core/kernels/ops_testutil.h"
#include "machina/core/kernels/ops_util.h"
#include "machina/core/lib/core/status_test_util.h"
#include "machina/core/platform/test.h"

namespace machina {

class AdjustContrastOpTest : public OpsTestBase {};

TEST_F(AdjustContrastOpTest, Simple_1113) {
  TF_EXPECT_OK(NodeDefBuilder("adjust_contrast_op", "AdjustContrastv2")
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());
  AddInputFromArray<float>(TensorShape({1, 1, 1, 3}), {-1, 2, 3});
  AddInputFromArray<float>(TensorShape({}), {1.0});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 1, 1, 3}));
  test::FillValues<float>(&expected, {-1, 2, 3});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(AdjustContrastOpTest, Simple_1223) {
  TF_EXPECT_OK(NodeDefBuilder("adjust_contrast_op", "AdjustContrastv2")
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());
  AddInputFromArray<float>(TensorShape({1, 2, 2, 3}),
                           {1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12});
  AddInputFromArray<float>(TensorShape({}), {0.2f});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 2, 2, 3}));
  test::FillValues<float>(&expected, {2.2, 6.2, 10.2, 2.4, 6.4, 10.4, 2.6, 6.6,
                                      10.6, 2.8, 6.8, 10.8});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(AdjustContrastOpTest, Big_99x99x3) {
  TF_EXPECT_OK(NodeDefBuilder("adjust_contrast_op", "AdjustContrastv2")
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());

  std::vector<float> values;
  values.reserve(99 * 99 * 3);
  for (int i = 0; i < 99 * 99 * 3; ++i) {
    values.push_back(i % 255);
  }

  AddInputFromArray<float>(TensorShape({1, 99, 99, 3}), values);
  AddInputFromArray<float>(TensorShape({}), {0.2f});
  TF_ASSERT_OK(RunOpKernel());
}

}  // namespace machina
