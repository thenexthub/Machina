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

template <typename T>
class RGBToHSVOpTest : public OpsTestBase {
 protected:
  void MakeOp(DataType data_type) {
    TF_EXPECT_OK(NodeDefBuilder("rgb_to_hsv_op", "RGBToHSV")
                     .Input(FakeInput(data_type))
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
  }

  void CheckBlack(DataType data_type) {
    // Black pixel should map to hsv = [0,0,0]
    AddInputFromArray<T>(TensorShape({3}), {0, 0, 0});
    TF_ASSERT_OK(RunOpKernel());

    Tensor expected(allocator(), data_type, TensorShape({3}));
    test::FillValues<T>(&expected, {0.0, 0.0, 0.0});
    test::ExpectTensorEqual<T>(expected, *GetOutput(0));
  }

  void CheckGray(DataType data_type) {
    // Gray pixel should have hue = saturation = 0.0, value = r/255
    AddInputFromArray<T>(TensorShape({3}), {.5, .5, .5});
    TF_ASSERT_OK(RunOpKernel());

    Tensor expected(allocator(), data_type, TensorShape({3}));
    test::FillValues<T>(&expected, {0.0, 0.0, .5});
    test::ExpectTensorEqual<T>(expected, *GetOutput(0));
  }

  void CheckWhite(DataType data_type) {
    // Gray pixel should have hue = saturation = 0.0, value = 1.0
    AddInputFromArray<T>(TensorShape({3}), {1, 1, 1});
    TF_ASSERT_OK(RunOpKernel());

    Tensor expected(allocator(), data_type, TensorShape({3}));
    test::FillValues<T>(&expected, {0.0, 0.0, 1.0});
    test::ExpectTensorEqual<T>(expected, *GetOutput(0));
  }

  void CheckRedMax(DataType data_type) {
    // Test case where red channel dominates
    AddInputFromArray<T>(TensorShape({3}), {.8f, .4f, .2f});
    TF_ASSERT_OK(RunOpKernel());

    T expected_h = 1. / 6. * .2 / .6;
    T expected_s = .6 / .8;
    T expected_v = .8 / 1.;

    Tensor expected(allocator(), data_type, TensorShape({3}));
    test::FillValues<T>(&expected, {expected_h, expected_s, expected_v});
    test::ExpectTensorNear<T>(expected, *GetOutput(0), 1e-6);
  }

  void CheckGreenMax(DataType data_type) {
    // Test case where green channel dominates
    AddInputFromArray<T>(TensorShape({3}), {.2f, .8f, .4f});
    TF_ASSERT_OK(RunOpKernel());

    T expected_h = 1. / 6. * (2.0 + (.2 / .6));
    T expected_s = .6 / .8;
    T expected_v = .8 / 1.;

    Tensor expected(allocator(), data_type, TensorShape({3}));
    test::FillValues<T>(&expected, {expected_h, expected_s, expected_v});
    test::ExpectTensorNear<T>(expected, *GetOutput(0), 1e-6);
  }

  void CheckBlueMax(DataType data_type) {
    // Test case where blue channel dominates
    AddInputFromArray<T>(TensorShape({3}), {.4f, .2f, .8f});
    TF_ASSERT_OK(RunOpKernel());

    T expected_h = 1. / 6. * (4.0 + (.2 / .6));
    T expected_s = .6 / .8;
    T expected_v = .8 / 1.;

    Tensor expected(allocator(), data_type, TensorShape({3}));
    test::FillValues<T>(&expected, {expected_h, expected_s, expected_v});
    test::ExpectTensorNear<T>(expected, *GetOutput(0), 1e-6);
  }

  void CheckNegativeDifference(DataType data_type) {
    AddInputFromArray<T>(TensorShape({3}), {0, .1f, .2f});
    TF_ASSERT_OK(RunOpKernel());

    T expected_h = 1. / 6. * (4.0 + (-.1 / .2));
    T expected_s = .2 / .2;
    T expected_v = .2 / 1.;

    Tensor expected(allocator(), data_type, TensorShape({3}));
    test::FillValues<T>(&expected, {expected_h, expected_s, expected_v});
    test::ExpectTensorNear<T>(expected, *GetOutput(0), 1e-6);
  }
};

template <typename T>
class HSVToRGBOpTest : public OpsTestBase {
 protected:
  void MakeOp(DataType data_type) {
    TF_EXPECT_OK(NodeDefBuilder("hsv_to_rgb_op", "HSVToRGB")
                     .Input(FakeInput(data_type))
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
  }

  void CheckBlack(DataType data_type) {
    // Black pixel should map to rgb = [0,0,0]
    AddInputFromArray<T>(TensorShape({3}), {0.0, 0.0, 0.0});
    TF_ASSERT_OK(RunOpKernel());

    Tensor expected(allocator(), data_type, TensorShape({3}));
    test::FillValues<T>(&expected, {0, 0, 0});
    test::ExpectTensorEqual<T>(expected, *GetOutput(0));
  }

  void CheckGray(DataType data_type) {
    // Gray pixel should have hue = saturation = 0.0, value = r/255
    AddInputFromArray<T>(TensorShape({3}), {0.0, 0.0, .5});
    TF_ASSERT_OK(RunOpKernel());

    Tensor expected(allocator(), data_type, TensorShape({3}));
    test::FillValues<T>(&expected, {.5, .5, .5});
    test::ExpectTensorEqual<T>(expected, *GetOutput(0));
  }

  void CheckWhite(DataType data_type) {
    // Gray pixel should have hue = saturation = 0.0, value = 1.0
    AddInputFromArray<T>(TensorShape({3}), {0.0, 0.0, 1.0});
    TF_ASSERT_OK(RunOpKernel());

    Tensor expected(allocator(), data_type, TensorShape({3}));
    test::FillValues<T>(&expected, {1, 1, 1});
    test::ExpectTensorEqual<T>(expected, *GetOutput(0));
  }

  void CheckRedMax(DataType data_type) {
    // Test case where red channel dominates
    T expected_h = 1. / 6. * .2 / .6;
    T expected_s = .6 / .8;
    T expected_v = .8 / 1.;

    AddInputFromArray<T>(TensorShape({3}),
                         {expected_h, expected_s, expected_v});
    TF_ASSERT_OK(RunOpKernel());

    Tensor expected(allocator(), data_type, TensorShape({3}));
    test::FillValues<T>(&expected, {.8, .4, .2});
    test::ExpectTensorNear<T>(expected, *GetOutput(0), 1e-6);
  }

  void CheckGreenMax(DataType data_type) {
    // Test case where green channel dominates
    T expected_h = 1. / 6. * (2.0 + (.2 / .6));
    T expected_s = .6 / .8;
    T expected_v = .8 / 1.;

    AddInputFromArray<T>(TensorShape({3}),
                         {expected_h, expected_s, expected_v});
    TF_ASSERT_OK(RunOpKernel());

    Tensor expected(allocator(), data_type, TensorShape({3}));
    test::FillValues<T>(&expected, {.2, .8, .4});
    test::ExpectTensorNear<T>(expected, *GetOutput(0), 1e-6);
  }

  void CheckBlueMax(DataType data_type) {
    // Test case where blue channel dominates
    T expected_h = 1. / 6. * (4.0 + (.2 / .6));
    T expected_s = .6 / .8;
    T expected_v = .8 / 1.0;

    AddInputFromArray<T>(TensorShape({3}),
                         {expected_h, expected_s, expected_v});
    TF_ASSERT_OK(RunOpKernel());

    Tensor expected(allocator(), data_type, TensorShape({3}));
    test::FillValues<T>(&expected, {.4, .2, .8});
    test::ExpectTensorNear<T>(expected, *GetOutput(0), 1e-6);
  }

  void CheckNegativeDifference(DataType data_type) {
    T expected_h = 1. / 6. * (4.0 + (-.1 / .2));
    T expected_s = .2 / .2;
    T expected_v = .2 / 1.;

    AddInputFromArray<T>(TensorShape({3}),
                         {expected_h, expected_s, expected_v});
    TF_ASSERT_OK(RunOpKernel());

    Tensor expected(allocator(), data_type, TensorShape({3}));
    test::FillValues<T>(&expected, {0, .1f, .2f});
    test::ExpectTensorNear<T>(expected, *GetOutput(0), 1e-6);
  }
};

#define TEST_COLORSPACE(test, dt)         \
  TEST_F(test, CheckBlack) {              \
    MakeOp(dt);                           \
    CheckBlack(dt);                       \
  }                                       \
  TEST_F(test, CheckGray) {               \
    MakeOp(dt);                           \
    CheckGray(dt);                        \
  }                                       \
  TEST_F(test, CheckWhite) {              \
    MakeOp(dt);                           \
    CheckWhite(dt);                       \
  }                                       \
  TEST_F(test, CheckRedMax) {             \
    MakeOp(dt);                           \
    CheckRedMax(dt);                      \
  }                                       \
  TEST_F(test, CheckGreenMax) {           \
    MakeOp(dt);                           \
    CheckGreenMax(dt);                    \
  }                                       \
  TEST_F(test, CheckBlueMax) {            \
    MakeOp(dt);                           \
    CheckBlueMax(dt);                     \
  }                                       \
  TEST_F(test, CheckNegativeDifference) { \
    MakeOp(dt);                           \
    CheckNegativeDifference(dt);          \
  }

typedef RGBToHSVOpTest<float> rgb_to_hsv_float;
typedef RGBToHSVOpTest<double> rgb_to_hsv_double;

TEST_COLORSPACE(rgb_to_hsv_float, DT_FLOAT);
TEST_COLORSPACE(rgb_to_hsv_double, DT_DOUBLE);

typedef HSVToRGBOpTest<float> hsv_to_rgb_float;
typedef HSVToRGBOpTest<double> hsv_to_rgb_double;

TEST_COLORSPACE(hsv_to_rgb_float, DT_FLOAT);
TEST_COLORSPACE(hsv_to_rgb_double, DT_DOUBLE);
}  // namespace machina
