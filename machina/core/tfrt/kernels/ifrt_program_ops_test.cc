/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Friday, August 8, 2025.
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
#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "machina/xla/python/ifrt/test_util.h"
#include "machina/xla/tsl/framework/serving_device_selector.h"
#include "machina/xla/tsl/framework/test_util/mock_serving_device_selector.h"
#include "machina/xla/tsl/lib/core/status_test_util.h"
#include "machina/xla/tsl/platform/status.h"
#include "machina/xla/tsl/platform/statusor.h"
#include "machina/core/framework/fake_input.h"
#include "machina/core/framework/node_def_builder.h"
#include "machina/core/framework/tensor.h"
#include "machina/core/framework/tensor_matcher.h"
#include "machina/core/framework/tensor_shape.h"
#include "machina/core/framework/tensor_testutil.h"
#include "machina/core/framework/types.pb.h"
#include "machina/core/kernels/ops_testutil.h"
#include "machina/core/lib/gtl/cleanup.h"
#include "machina/core/platform/status.h"
#include "machina/core/platform/test.h"
#include "machina/core/tfrt/ifrt/ifrt_executable_registry.h"
#include "machina/core/tfrt/ifrt/ifrt_serving_executable_test_util.h"

namespace machina {
namespace tfrt_stub {
namespace {

using machina::ifrt_serving::ServingExecutableRegistry;
using machina::ifrt_serving::test_utils::GetMlirModulePath;
using machina::ifrt_serving::test_utils::IfrtServingExecutableTestHelper;
using machina::test::AsTensor;
using machina::test::TensorEq;
using ::testing::Return;

class IfrtCallOpTest : public OpsTestBase {
 protected:
  absl::Status Init(int64_t program_id, int num_inputs, DataType input_type,
                    const std::vector<int>& variable_arg_indices,
                    const std::vector<DataType>& output_type_list) {
    TF_CHECK_OK(NodeDefBuilder("op", "IfrtCall")
                    .Input(FakeInput(num_inputs, input_type))
                    .Attr("program_id", program_id)
                    .Attr("variable_arg_indices", variable_arg_indices)
                    .Attr("Tout", output_type_list)
                    .Finalize(node_def()));
    return InitOp();
  }
};

TEST_F(IfrtCallOpTest, Basic) {
  int64_t program_id = 123;
  TF_ASSERT_OK(Init(
      /*program_id=*/program_id,
      /*num_inputs=*/2,
      /*input_type=*/DT_INT32,
      /*variable_arg_indices=*/{},
      /*output_type_list=*/{DT_INT32}));

  tsl::test_util::MockServingDeviceSelector selector;
  IfrtServingExecutableTestHelper helper(&selector);
  EXPECT_CALL(selector, ReserveDevice(absl::StrCat(program_id)))
      .Times(1)
      .WillOnce(Return(tsl::DeviceReservation(0, /*selector=*/nullptr)));
  auto executable =
      helper.MakeExecutable(program_id, GetMlirModulePath("executable.mlir"));

  TF_ASSERT_OK_AND_ASSIGN(
      ServingExecutableRegistry::Handle handle,
      ServingExecutableRegistry::Register(program_id, std::move(executable)));
  auto handle_cleaner = gtl::MakeCleanup([&handle] { handle.Release(); });

  AddInputFromArray<int32_t>(TensorShape({1, 3}), {1, 2, 3});
  AddInputFromArray<int32_t>(TensorShape({3, 1}), {1, 2, 3});
  // Run warmup execution plus one for core selection.
  for (int i = 0; i < helper.num_cores() + 1; ++i) {
    TF_ASSERT_OK(RunOpKernel());
  }
  Tensor expected_out = AsTensor<int32_t>({14}, TensorShape({1, 1}));
  EXPECT_THAT(*GetOutput(0), TensorEq(expected_out));
}

}  // namespace
}  // namespace tfrt_stub
}  // namespace machina
