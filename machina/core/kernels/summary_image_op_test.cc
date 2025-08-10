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

#include "machina/core/framework/allocator.h"
#include "machina/core/framework/fake_input.h"
#include "machina/core/framework/node_def_builder.h"
#include "machina/core/framework/op_kernel.h"
#include "machina/core/framework/summary.pb.h"
#include "machina/core/framework/tensor.h"
#include "machina/core/framework/types.h"
#include "machina/core/kernels/ops_testutil.h"
#include "machina/core/kernels/ops_util.h"
#include "machina/core/lib/core/status_test_util.h"
#include "machina/core/lib/histogram/histogram.h"
#include "machina/core/lib/strings/strcat.h"
#include "machina/core/platform/env.h"
#include "machina/core/platform/logging.h"
#include "machina/core/platform/protobuf.h"
#include "machina/core/platform/test.h"

namespace machina {
namespace {

static void EXPECT_SummaryMatches(const Summary& actual,
                                  const string& expected_str) {
  Summary expected;
  CHECK(protobuf::TextFormat::ParseFromString(expected_str, &expected));
  EXPECT_EQ(expected.DebugString(), actual.DebugString());
}

// --------------------------------------------------------------------------
// SummaryImageOp
// --------------------------------------------------------------------------
class SummaryImageOpTest : public OpsTestBase {
 protected:
  void MakeOp(int max_images) {
    TF_ASSERT_OK(NodeDefBuilder("myop", "ImageSummary")
                     .Input(FakeInput())
                     .Input(FakeInput())
                     .Attr("max_images", max_images)
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }

  void CheckAndRemoveEncodedImages(Summary* summary) {
    for (int i = 0; i < summary->value_size(); ++i) {
      Summary::Value* value = summary->mutable_value(i);
      ASSERT_TRUE(value->has_image()) << "No image for value: " << value->tag();
      ASSERT_FALSE(value->image().encoded_image_string().empty())
          << "No encoded_image_string for value: " << value->tag();
      if (VLOG_IS_ON(2)) {
        // When LOGGING, output the images to disk for manual inspection.
        TF_CHECK_OK(WriteStringToFile(
            Env::Default(), strings::StrCat("/tmp/", value->tag(), ".png"),
            value->image().encoded_image_string()));
      }
      value->mutable_image()->clear_encoded_image_string();
    }
  }
};

TEST_F(SummaryImageOpTest, ThreeGrayImagesOutOfFive4dInput) {
  MakeOp(3 /* max images */);

  // Feed and run
  AddInputFromArray<tstring>(TensorShape({}), {"tag"});
  AddInputFromArray<float>(TensorShape({5, 2, 1, 1}),
                           {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output size.
  Tensor* out_tensor = GetOutput(0);
  ASSERT_EQ(0, out_tensor->dims());
  Summary summary;
  ParseProtoUnlimited(&summary, out_tensor->scalar<tstring>()());

  CheckAndRemoveEncodedImages(&summary);
  EXPECT_SummaryMatches(summary, R"(
    value { tag: 'tag/image/0' image { width: 1 height: 2 colorspace: 1} }
    value { tag: 'tag/image/1' image { width: 1 height: 2 colorspace: 1} }
    value { tag: 'tag/image/2' image { width: 1 height: 2 colorspace: 1} }
  )");
}

TEST_F(SummaryImageOpTest, OneGrayImage4dInput) {
  MakeOp(1 /* max images */);

  // Feed and run
  AddInputFromArray<tstring>(TensorShape({}), {"tag"});
  AddInputFromArray<float>(TensorShape({5 /*batch*/, 2, 1, 1 /*depth*/}),
                           {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output size.
  Tensor* out_tensor = GetOutput(0);
  ASSERT_EQ(0, out_tensor->dims());
  Summary summary;
  ParseProtoUnlimited(&summary, out_tensor->scalar<tstring>()());

  CheckAndRemoveEncodedImages(&summary);
  EXPECT_SummaryMatches(summary, R"(
    value { tag: 'tag/image' image { width: 1 height: 2 colorspace: 1} })");
}

TEST_F(SummaryImageOpTest, OneColorImage4dInput) {
  MakeOp(1 /* max images */);

  // Feed and run
  AddInputFromArray<tstring>(TensorShape({}), {"tag"});
  AddInputFromArray<float>(
      TensorShape({1 /*batch*/, 5 /*rows*/, 2 /*columns*/, 3 /*depth*/}),
      {
          /* r0, c0, RGB */ 1.0f, 0.1f, 0.2f,
          /* r0, c1, RGB */ 1.0f, 0.3f, 0.4f,
          /* r1, c0, RGB */ 0.0f, 1.0f, 0.0f,
          /* r1, c1, RGB */ 0.0f, 1.0f, 0.0f,
          /* r2, c0, RGB */ 0.0f, 0.0f, 1.0f,
          /* r2, c1, RGB */ 0.0f, 0.0f, 1.0f,
          /* r3, c0, RGB */ 1.0f, 1.0f, 0.0f,
          /* r3, c1, RGB */ 1.0f, 0.0f, 1.0f,
          /* r4, c0, RGB */ 1.0f, 1.0f, 0.0f,
          /* r4, c1, RGB */ 1.0f, 0.0f, 1.0f,
      });
  TF_ASSERT_OK(RunOpKernel());

  // Check the output size.
  Tensor* out_tensor = GetOutput(0);
  ASSERT_EQ(0, out_tensor->dims());
  Summary summary;
  ParseProtoUnlimited(&summary, out_tensor->scalar<tstring>()());

  CheckAndRemoveEncodedImages(&summary);
  EXPECT_SummaryMatches(summary, R"(
    value { tag: 'tag/image' image { width: 2 height: 5 colorspace: 3} })");
}

}  // namespace
}  // namespace machina
