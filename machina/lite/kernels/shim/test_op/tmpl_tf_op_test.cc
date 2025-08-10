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
#include <cstdint>

#include <gtest/gtest.h>
#include "machina/xla/tsl/lib/core/status_test_util.h"
#include "machina/core/framework/fake_input.h"
#include "machina/core/framework/node_def_builder.h"
#include "machina/core/framework/tensor_shape.h"
#include "machina/core/framework/tensor_testutil.h"
#include "machina/core/framework/types.pb.h"
#include "machina/core/kernels/ops_testutil.h"

namespace tflite {
namespace shim {
namespace {

using ::machina::DT_FLOAT;
using ::machina::DT_INT32;
using ::machina::DT_INT64;
using ::machina::FakeInput;
using ::machina::NodeDefBuilder;
using ::machina::TensorShape;
using ::machina::test::AsTensor;
using ::machina::test::ExpectTensorEqual;

class TmplOpTfTest : public ::machina::OpsTestBase {};

TEST_F(TmplOpTfTest, float_int32) {
  // Prepare graph.
  TF_ASSERT_OK(NodeDefBuilder("tmpl_op", "TemplatizedOperation")
                   .Attr("AType", DT_FLOAT)
                   .Attr("BType", DT_INT32)
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_INT32))
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  AddInputFromArray<float>(TensorShape({}), {10.5});
  AddInputFromArray<int32_t>(TensorShape({}), {20});

  TF_ASSERT_OK(RunOpKernel());

  // Validate the output.
  ExpectTensorEqual<float>(*GetOutput(0),
                           AsTensor<float>({30.5}, /*shape=*/{}));
}

TEST_F(TmplOpTfTest, int32_int64) {
  // Prepare graph.
  TF_ASSERT_OK(NodeDefBuilder("tmpl_op", "TemplatizedOperation")
                   .Attr("AType", DT_INT32)
                   .Attr("BType", DT_INT64)
                   .Input(FakeInput(DT_INT32))
                   .Input(FakeInput(DT_INT64))
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  AddInputFromArray<int32_t>(TensorShape({}), {10});
  AddInputFromArray<int64_t>(TensorShape({}), {20});

  TF_ASSERT_OK(RunOpKernel());

  // Validate the output.
  ExpectTensorEqual<float>(*GetOutput(0), AsTensor<float>({30}, /*shape=*/{}));
}

}  // namespace
}  // namespace shim
}  // namespace tflite
