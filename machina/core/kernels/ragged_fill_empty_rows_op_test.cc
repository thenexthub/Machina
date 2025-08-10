/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, June 15, 2025.
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

#include <gtest/gtest.h>
#include "machina/core/framework/fake_input.h"
#include "machina/core/framework/node_def_builder.h"
#include "machina/core/framework/shape_inference.h"
#include "machina/core/framework/shape_inference_testutil.h"
#include "machina/core/framework/tensor.h"
#include "machina/core/framework/tensor_shape.h"
#include "machina/core/framework/tensor_testutil.h"
#include "machina/core/kernels/ops_testutil.h"
#include "machina/core/lib/core/status_test_util.h"
#include "machina/core/platform/test.h"

namespace machina {
namespace {

class RaggedFillEmptyRowsOpTest : public ::machina::OpsTestBase {
 protected:
  const int kValueRowidsOutput = 0;
  const int kValuesOutput = 1;
  const int kEmptyRowIndicatorOutput = 2;
  const int kReverseIndexMapOutput = 3;

  // Builds the machina test graph for the RaggedFillEmptyRows op.
  template <typename T>
  void BuildFillEmptyRowsGraph() {
    const auto& dtype = DataTypeToEnum<T>::v();
    const auto& dtype_int64 = DataTypeToEnum<int64_t>::v();
    TF_ASSERT_OK(NodeDefBuilder("tested_op", "RaggedFillEmptyRows")
                     .Input(FakeInput(dtype_int64))  // value_rowids
                     .Input(FakeInput(dtype))        // values
                     .Input(FakeInput(dtype_int64))  // nrows
                     .Input(FakeInput(dtype))        // default value
                     .Attr("T", dtype)
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

TEST_F(RaggedFillEmptyRowsOpTest, IntValues) {
  BuildFillEmptyRowsGraph<int>();
  AddInputFromArray<int64_t>(TensorShape({4}), {1, 2, 2, 5});  // value_rowids
  AddInputFromArray<int>(TensorShape({4}), {2, 4, 6, 8});      // values
  AddInputFromArray<int64_t>(TensorShape({}), {7});            // nrows
  AddInputFromArray<int>(TensorShape({}), {-1});               // default value
  TF_ASSERT_OK(RunOpKernel());

  test::ExpectTensorEqual<int64_t>(
      *GetOutput(kValueRowidsOutput),
      test::AsTensor<int64_t>({0, 1, 2, 2, 3, 4, 5, 6}));
  test::ExpectTensorEqual<int>(
      *GetOutput(kValuesOutput),
      test::AsTensor<int>({-1, 2, 4, 6, -1, -1, 8, -1}));
}

TEST_F(RaggedFillEmptyRowsOpTest, FloatValues) {
  BuildFillEmptyRowsGraph<float>();
  AddInputFromArray<int64_t>(TensorShape({4}), {1, 2, 2, 5});    // value_rowids
  AddInputFromArray<float>(TensorShape({4}), {2., 4., 6., 8.});  // values
  AddInputFromArray<int64_t>(TensorShape({}), {7});              // nrows
  AddInputFromArray<float>(TensorShape({}), {-1.});  // default value
  TF_ASSERT_OK(RunOpKernel());

  test::ExpectTensorEqual<int64_t>(
      *GetOutput(kValueRowidsOutput),
      test::AsTensor<int64_t>({0, 1, 2, 2, 3, 4, 5, 6}));
  test::ExpectTensorEqual<float>(
      *GetOutput(kValuesOutput),
      test::AsTensor<float>({-1., 2., 4., 6., -1., -1., 8., -1.}));
}

}  // namespace
}  // namespace machina
