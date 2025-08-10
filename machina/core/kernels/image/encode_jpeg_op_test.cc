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

#include "machina/core/framework/fake_input.h"
#include "machina/core/framework/node_def_builder.h"
#include "machina/core/framework/op_kernel.h"
#include "machina/core/framework/tensor.h"
#include "machina/core/framework/types.h"
#include "machina/core/graph/node_builder.h"
#include "machina/core/kernels/ops_testutil.h"
#include "machina/core/lib/core/status_test_util.h"
#include "machina/core/platform/test.h"

namespace machina {
namespace {

using EncodeJpegWithVariableQualityTest = OpsTestBase;

TEST_F(EncodeJpegWithVariableQualityTest, FailsForInvalidQuality) {
  TF_ASSERT_OK(NodeDefBuilder("encode_op", "EncodeJpegVariableQuality")
                   .Input(FakeInput(DT_UINT8))
                   .Input(FakeInput(DT_INT32))
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  AddInputFromArray<uint8>(TensorShape({2, 2, 3}),
                           {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
  AddInputFromArray<int32>(TensorShape({}), {200});
  absl::Status status = RunOpKernel();
  EXPECT_TRUE(absl::IsInvalidArgument(status));
  EXPECT_TRUE(absl::StartsWith(status.message(), "quality must be in [0,100]"));
}

}  // namespace
}  // namespace machina
