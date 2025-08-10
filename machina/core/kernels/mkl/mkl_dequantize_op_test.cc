/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, May 15, 2025.
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

#if defined(INTEL_MKL)
#define EIGEN_USE_THREADS

#include "machina/cc/ops/const_op.h"
#include "machina/cc/ops/standard_ops.h"
#include "machina/core/framework/fake_input.h"
#include "machina/core/framework/node_def_builder.h"
#include "machina/core/framework/tensor.h"
#include "machina/core/kernels/mkl/mkl_kernel_util.h"
#include "machina/core/kernels/ops_testutil.h"
#include "machina/core/kernels/ops_util.h"
#include "machina/core/kernels/quantization_utils.h"
#include "machina/core/lib/core/status_test_util.h"
#include "machina/core/platform/test.h"
#include "machina/core/platform/test_benchmark.h"
#include "machina/core/util/mkl_util.h"

namespace machina {

class MklDequantizeOpTest : public OpsTestBase {
 protected:
  template <typename Tinput, typename Toutput>
  void RunMklDequantize(const Tensor& input_quantized,
                        const Tensor& min_range_float,
                        const Tensor& max_range_float,
                        const Tensor& expected_output) {
    AddInputFromArray<Tinput>(input_quantized.shape(),
                              input_quantized.flat<Tinput>());
    AddInputFromArray<float>(min_range_float.shape(),
                             min_range_float.flat<float>());
    AddInputFromArray<float>(max_range_float.shape(),
                             max_range_float.flat<float>());

    TF_ASSERT_OK(RunOpKernel());

    const Tensor& actual_output = *GetOutput(0);
    test::ExpectTensorNear<Toutput>(expected_output, actual_output, 0.1);
  }

  template <typename Tinput, typename Toutput>
  void TestMklDequantize() {
    const DataType input_dt = DataTypeToEnum<Tinput>::v();
    const DataType output_dt = DataTypeToEnum<Toutput>::v();

    TF_ASSERT_OK(NodeDefBuilder("dequantize_op", "_MklDequantize")
                     .Input(FakeInput(input_dt))
                     .Input(FakeInput(DT_FLOAT))  // min_range
                     .Input(FakeInput(DT_FLOAT))  // max_range
                     .Attr("T", input_dt)
                     .Attr("dtype", output_dt)
                     .Attr("mode", "SCALED")
                     .Attr("_kernel", "QuantizedMklOp")
                     .Finalize(node_def()));

    TF_ASSERT_OK(InitOp());

    Tensor input_float(DT_FLOAT, {1, 2, 2, 2});
    test::FillValues<float>(&input_float, {0, 10, 50, 40, 25, 115, 190, 255});

    const float min_range = 0.0f;
    const float max_range = 255.0f;

    Tensor min_range_float(DT_FLOAT, {});
    test::FillValues<float>(&min_range_float, {min_range});

    Tensor max_range_float(DT_FLOAT, {});
    test::FillValues<float>(&max_range_float, {max_range});

    Tensor input_quantized =
        FloatTensorToQuantized<Tinput>(input_float, min_range, max_range);

    Tensor expected_output_float32;
    MklTestingUtil::RunDequantizeOp(input_quantized, min_range_float,
                                    max_range_float, "SCALED",
                                    &expected_output_float32);

    if (output_dt == DT_BFLOAT16) {
      // Since DequantizeOp does not support "SCALED" mode for bf16 output,
      // use a workaround by casting fp32 output (computed using "SCALED" mode)
      // into bf16 output.
      Tensor expected_output_bfloat16(DT_BFLOAT16, {1, 2, 2, 2});
      expected_output_bfloat16.flat<bfloat16>() =
          expected_output_float32.flat<float>().cast<bfloat16>();
      RunMklDequantize<Tinput, Toutput>(input_quantized, min_range_float,
                                        max_range_float,
                                        expected_output_bfloat16);
    } else {
      RunMklDequantize<Tinput, Toutput>(input_quantized, min_range_float,
                                        max_range_float,
                                        expected_output_float32);
    }
  }
};

TEST_F(MklDequantizeOpTest, MklDequantize_Unsigned_Input_Float_Output) {
  TestMklDequantize<quint8, float>();
}

TEST_F(MklDequantizeOpTest, MklDequantize_Signed_Input_Float_Output) {
  TestMklDequantize<qint8, float>();
}

TEST_F(MklDequantizeOpTest, MklDequantize_Unsigned_Input_Bfloat16_Output) {
  TestMklDequantize<quint8, bfloat16>();
}

TEST_F(MklDequantizeOpTest, MklDequantize_Signed_Input_Bfloat16_Output) {
  TestMklDequantize<qint8, bfloat16>();
}

}  // namespace machina

#endif  // INTEL_MKL
