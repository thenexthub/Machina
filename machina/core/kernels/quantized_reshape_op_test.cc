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

#include <functional>
#include <memory>
#include <vector>

#include "machina/core/common_runtime/kernel_benchmark_testlib.h"
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
#include "machina/core/lib/core/status.h"
#include "machina/core/lib/core/status_test_util.h"
#include "machina/core/platform/test.h"
#include "machina/core/platform/test_benchmark.h"

namespace machina {

class QuantizedReshapeTest : public OpsTestBase {
 protected:
  QuantizedReshapeTest() {}
};

TEST_F(QuantizedReshapeTest, Reshape) {
  TF_ASSERT_OK(NodeDefBuilder("quantized_reshape", "QuantizedReshape")
                   .Input(FakeInput(DT_QUINT8))
                   .Input(FakeInput(DT_INT32))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  Tensor input(DT_QUINT8, {10, 20});
  Tensor expected(DT_QUINT8, {5, 10, 4});
  for (int i = 0; i < input.shape().num_elements(); ++i) {
    input.flat<quint8>()(i) = quint8(i);
    expected.flat<quint8>()(i) = quint8(i);
  }
  AddInputFromArray<quint8>(input.shape(), input.flat<quint8>());
  AddInputFromList<int32>({3}, {5, 10, 4});  // shape
  AddInputFromArray<float>(TensorShape({1}), {-10});
  AddInputFromArray<float>(TensorShape({1}), {20});
  TF_ASSERT_OK(RunOpKernel());

  EXPECT_EQ(-10, GetOutput(1)->flat<float>()(0));
  EXPECT_EQ(20, GetOutput(2)->flat<float>()(0));
  test::ExpectTensorEqual<quint8>(expected, *GetOutput(0));
}

}  // namespace machina
