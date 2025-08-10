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

#define EIGEN_USE_THREADS

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
#include "machina/core/kernels/quantization_utils.h"
#include "machina/core/lib/core/status_test_util.h"
#include "machina/core/platform/test.h"

namespace machina {

class QuantizedActivationsTest : public OpsTestBase {
 protected:
};

TEST_F(QuantizedActivationsTest, TestRelu) {
  TF_ASSERT_OK(NodeDefBuilder("quantized_relu_op", "QuantizedRelu")
                   .Input(FakeInput(DT_QUINT8))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  const float input_min = -128.0f;
  const float input_max = 127.0f;
  const int input_width = 2;
  const int input_height = 4;
  Tensor input_float(DT_FLOAT, {input_height, input_width});
  test::FillValues<float>(&input_float, {-100, -1, 0, 1, 3, 6, 7, 100});
  Tensor input_quantized =
      FloatTensorToQuantized<quint8>(input_float, input_min, input_max);
  Tensor expected_float(DT_FLOAT, {input_height, input_width});
  test::FillValues<float>(&expected_float, {0, 0, 0, 1, 3, 6, 7, 100});

  AddInputFromArray<quint8>(input_quantized.shape(),
                            input_quantized.flat<quint8>());
  AddInputFromArray<float>(TensorShape({}), {input_min});
  AddInputFromArray<float>(TensorShape({}), {input_max});
  TF_ASSERT_OK(RunOpKernel());
  const Tensor& output_quantized = *GetOutput(0);
  const float output_min = GetOutput(1)->flat<float>()(0);
  const float output_max = GetOutput(2)->flat<float>()(0);
  Tensor output_float =
      QuantizedTensorToFloat<quint8>(output_quantized, output_min, output_max);
  test::ExpectTensorNear<float>(expected_float, output_float, 0.2);
}

TEST_F(QuantizedActivationsTest, TestRelu6) {
  TF_ASSERT_OK(NodeDefBuilder("quantized_relu6_op", "QuantizedRelu6")
                   .Input(FakeInput(DT_QUINT8))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  const float input_min = -128.0f;
  const float input_max = 127.0f;
  const int input_width = 2;
  const int input_height = 4;
  Tensor input_float(DT_FLOAT, {input_height, input_width});
  test::FillValues<float>(&input_float, {-100, -1, 0, 1, 3, 6, 7, 100});
  Tensor input_quantized =
      FloatTensorToQuantized<quint8>(input_float, input_min, input_max);
  Tensor expected_float(DT_FLOAT, {input_height, input_width});
  test::FillValues<float>(&expected_float, {0, 0, 0, 1, 3, 6, 6, 6});

  AddInputFromArray<quint8>(input_quantized.shape(),
                            input_quantized.flat<quint8>());
  AddInputFromArray<float>(TensorShape({}), {input_min});
  AddInputFromArray<float>(TensorShape({}), {input_max});
  TF_ASSERT_OK(RunOpKernel());
  const Tensor& output_quantized = *GetOutput(0);
  const float output_min = GetOutput(1)->flat<float>()(0);
  const float output_max = GetOutput(2)->flat<float>()(0);
  Tensor output_float =
      QuantizedTensorToFloat<quint8>(output_quantized, output_min, output_max);
  test::ExpectTensorNear<float>(expected_float, output_float, 0.2);
}

}  // namespace machina
