/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 8, 2025.
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

#include <functional>
#include <memory>
#include <vector>

#include "machina/cc/client/client_session.h"
#include "machina/cc/ops/audio_ops.h"
#include "machina/cc/ops/const_op.h"
#include "machina/cc/ops/math_ops.h"
#include "machina/core/framework/tensor_testutil.h"
#include "machina/core/framework/types.h"
#include "machina/core/framework/types.pb.h"
#include "machina/core/kernels/ops_util.h"
#include "machina/core/lib/core/status_test_util.h"
#include "machina/core/platform/test.h"

namespace machina {
namespace ops {
namespace {

TEST(MfccOpTest, SimpleTest) {
  Scope root = Scope::DisabledShapeInferenceScope();

  Tensor spectrogram_tensor(DT_FLOAT, TensorShape({1, 1, 513}));
  test::FillIota<float>(&spectrogram_tensor, 1.0f);

  Output spectrogram_const_op = Const(root.WithOpName("spectrogram_const_op"),
                                      Input::Initializer(spectrogram_tensor));

  Output sample_rate_const_op =
      Const(root.WithOpName("sample_rate_const_op"), 22050);

  Mfcc mfcc_op = Mfcc(root.WithOpName("mfcc_op"), spectrogram_const_op,
                      sample_rate_const_op);

  TF_ASSERT_OK(root.status());

  ClientSession session(root);
  std::vector<Tensor> outputs;

  TF_EXPECT_OK(
      session.Run(ClientSession::FeedType(), {mfcc_op.output}, &outputs));

  const Tensor& mfcc_tensor = outputs[0];

  EXPECT_EQ(3, mfcc_tensor.dims());
  EXPECT_EQ(13, mfcc_tensor.dim_size(2));
  EXPECT_EQ(1, mfcc_tensor.dim_size(1));
  EXPECT_EQ(1, mfcc_tensor.dim_size(0));

  test::ExpectTensorNear<float>(
      mfcc_tensor,
      test::AsTensor<float>(
          {29.13970072, -6.41568601, -0.61903012, -0.96778652, -0.26819878,
           -0.40907028, -0.15614748, -0.23203119, -0.10481487, -0.1543029,
           -0.0769791, -0.10806114, -0.06047613},
          TensorShape({1, 1, 13})),
      1e-3);
}

}  // namespace
}  // namespace ops
}  // namespace machina
