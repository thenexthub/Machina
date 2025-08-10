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

#include "machina/core/framework/allocator.h"
#include "machina/core/framework/fake_input.h"
#include "machina/core/framework/node_def_builder.h"
#include "machina/core/framework/op_kernel.h"
#include "machina/core/framework/tensor.h"
#include "machina/core/framework/tensor_testutil.h"
#include "machina/core/framework/types.h"
#include "machina/core/kernels/ops_testutil.h"
#include "machina/core/lib/core/status_test_util.h"
#include "machina/core/platform/test.h"

namespace machina {

namespace {

class SparseReduceSumOpTest : public OpsTestBase {
 protected:
  template <typename T>
  void MakeOp() {
    DataType value_type = machina::DataTypeToEnum<T>::value;
    TF_ASSERT_OK(NodeDefBuilder("sparse_reduce_sum", "SparseReduceSum")
                     .Input(FakeInput(DT_INT64))
                     .Input(FakeInput(value_type))
                     .Input(FakeInput(DT_INT64))
                     .Input(FakeInput(DT_INT32))
                     .Attr("T", value_type)
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

TEST_F(SparseReduceSumOpTest, SimpleReduce) {
  MakeOp<float>();

  // [    1]
  // [2    ]
  // [3   4]

  const auto indices_shape = TensorShape({4, 2});
  std::initializer_list<int64_t> in{0, 1, 1, 0, 2, 0, 2, 1};
  const absl::Span<const int64_t> indices(in);
  std::initializer_list<int64_t> sh{3, 2};
  const absl::Span<const int64_t> shape(sh);

  AddInputFromArray<int64_t>(indices_shape, indices);
  AddInputFromArray<float>(TensorShape({4}), {1, 2, 3, 4});
  AddInputFromArray<int64_t>(TensorShape({2}), shape);
  AddInputFromArray<int32>(TensorShape({1}), {0});  // reduction axes

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected_by_0(allocator(), DT_FLOAT, TensorShape({2}));
  test::FillValues<float>(&expected_by_0, {5, 5});
  test::ExpectTensorEqual<float>(expected_by_0, *GetOutput(0));
}

class SparseReduceSumSparseOpTest : public OpsTestBase {
 protected:
  template <typename T>
  void MakeOp() {
    DataType value_type = machina::DataTypeToEnum<T>::value;
    TF_ASSERT_OK(
        NodeDefBuilder("sparse_reduce_sum_sparse", "SparseReduceSumSparse")
            .Input(FakeInput(DT_INT64))
            .Input(FakeInput(value_type))
            .Input(FakeInput(DT_INT64))
            .Input(FakeInput(DT_INT32))
            .Attr("T", value_type)
            .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

TEST_F(SparseReduceSumSparseOpTest, SimpleReduce) {
  MakeOp<float>();

  // [    2]
  // [2    ]
  // [3   4]

  const auto indices_shape = TensorShape({4, 2});
  std::initializer_list<int64_t> in{0, 1, 1, 0, 2, 0, 2, 1};
  const absl::Span<const int64_t> indices(in);
  std::initializer_list<int64_t> sh{3, 2};
  const absl::Span<const int64_t> shape(sh);

  AddInputFromArray<int64_t>(indices_shape, indices);
  AddInputFromArray<float>(TensorShape({4}), {2, 2, 3, 4});
  AddInputFromArray<int64_t>(TensorShape({2}), shape);
  AddInputFromArray<int32>(TensorShape({1}), {0});  // reduction axes

  TF_ASSERT_OK(RunOpKernel());

  const int expected_nnz = 2;
  Tensor expected_indices(allocator(), DT_INT64,
                          TensorShape({expected_nnz, 1}));
  test::FillValues<int64_t>(&expected_indices, {0, 1});
  test::ExpectTensorEqual<int64_t>(expected_indices, *GetOutput(0));

  Tensor expected_values(allocator(), DT_FLOAT, TensorShape({2}));
  test::FillValues<float>(&expected_values, {5, 6});
  test::ExpectTensorEqual<float>(expected_values, *GetOutput(1));

  Tensor expected_shape(allocator(), DT_INT64, {1});
  test::FillValues<int64_t>(&expected_shape, {2});
  test::ExpectTensorEqual<int64_t>(expected_shape, *GetOutput(2));
}

}  // namespace

}  // namespace machina
