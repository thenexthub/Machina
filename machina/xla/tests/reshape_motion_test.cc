/* Copyright 2017 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <memory>
#include <numeric>
#include <random>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "machina/xla/array2d.h"
#include "machina/xla/array4d.h"
#include "machina/xla/client/local_client.h"
#include "machina/xla/hlo/builder/xla_builder.h"
#include "machina/xla/hlo/testlib/test_helpers.h"
#include "machina/xla/layout_util.h"
#include "machina/xla/literal.h"
#include "machina/xla/reference_util.h"
#include "machina/xla/shape_util.h"
#include "machina/xla/status_macros.h"
#include "machina/xla/tests/client_library_test_base.h"
#include "machina/xla/tests/literal_test_util.h"
#include "tsl/platform/test.h"

namespace xla {
namespace {

using ReshapeMotionTest = ClientLibraryTestBase;

TEST_F(ReshapeMotionTest, ElementwiseOfReshapesWithNonSameInputShapes) {
  XlaBuilder builder(TestName());
  auto a = ConstantR2<int32_t>(&builder, {{2, 3, 5}, {7, 11, 13}});
  auto b = ConstantR2<int32_t>(&builder, {{17, 19}, {23, 29}, {31, 37}});
  auto c = Reshape(a, {6});
  auto d = Reshape(b, {6});
  Mul(c, d);

  ComputeAndCompareR1<int32_t>(&builder, {34, 57, 115, 203, 341, 481}, {});
}

}  // namespace
}  // namespace xla
