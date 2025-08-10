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
#include "machina/core/tfrt/fallback/cost_recorder.h"

#include <cstdint>
#include <limits>
#include <string>

#include <gtest/gtest.h>
#include "machina/core/platform/env.h"
#include "machina/core/tfrt/fallback/op_cost_map.pb.h"

namespace machina {
namespace tfrt_stub {
namespace {

constexpr int64_t kTestOpKey = 1;
constexpr uint64_t kTestCost = 1234;
constexpr uint64_t kTestAvgCost = 1851;

TEST(CostRecorderTest, RecordCostTest) {
  CostRecorder recorder;

  recorder.RecordCost(kTestOpKey, kTestCost);
  recorder.RecordCost(kTestOpKey, kTestCost);

  EXPECT_EQ(recorder.size(), 1);
}

TEST(CostRecorderTest, GetCostTest) {
  CostRecorder recorder;

  recorder.RecordCost(kTestOpKey, kTestCost);
  recorder.RecordCost(kTestOpKey, 2 * kTestCost);

  EXPECT_EQ(recorder.size(), 1);
  EXPECT_EQ(recorder.GetCost(kTestOpKey), kTestAvgCost);
}

TEST(CostRecorderTest, GetCostDefaultValueTest) {
  CostRecorder recorder;
  ASSERT_EQ(recorder.size(), 0);

  EXPECT_EQ(recorder.GetCost(kTestOpKey),
            std::numeric_limits<uint32_t>::max());
}

TEST(CostRecorderTest, WriteToFileTest) {
  CostRecorder recorder;
  ASSERT_EQ(recorder.size(), 0);

  std::string measured_cost_path;
  machina::Env::Default()->LocalTempFilename(&measured_cost_path);
  ASSERT_EQ(setenv("TF_TFRT_MEASURED_COST_PATH", measured_cost_path.c_str(), 1),
            0);
  TF_CHECK_OK(recorder.WriteToFile());

  OpCostMapProto op_cost_map_proto;
  TF_CHECK_OK(machina::ReadTextProto(
      machina::Env::Default(), measured_cost_path, &op_cost_map_proto));

  EXPECT_EQ(op_cost_map_proto.op_cost_map_size(), 0);
}

TEST(CostRecorderTest, ProtoRecordsTest) {
  CostRecorder recorder;

  // Records the cost of op.
  recorder.RecordCost(kTestOpKey, kTestCost);
  recorder.RecordCost(kTestOpKey, 2 * kTestCost);
  ASSERT_EQ(recorder.size(), 1);

  // Writes op's cost to the disk.
  std::string measured_cost_path;
  machina::Env::Default()->LocalTempFilename(&measured_cost_path);
  ASSERT_EQ(setenv(CostRecorder::MesuredCostPathEnvVarName(),
                   measured_cost_path.c_str(), 1),
            0);
  TF_CHECK_OK(recorder.WriteToFile());

  // Reads op's cost from the disk.
  OpCostMapProto op_cost_map_proto;
  TF_CHECK_OK(machina::ReadTextProto(
      machina::Env::Default(), measured_cost_path, &op_cost_map_proto));

  EXPECT_EQ(op_cost_map_proto.op_cost_map().find(kTestOpKey)->second,
            kTestAvgCost);
}

}  // namespace
}  // namespace tfrt_stub
}  // namespace machina
