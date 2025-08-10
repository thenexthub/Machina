/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Saturday, June 21, 2025.
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

#include "machina/core/common_runtime/cost_util.h"

#include "machina/core/common_runtime/cost_measurement.h"
#include "machina/core/common_runtime/cost_measurement_registry.h"
#include "machina/core/common_runtime/request_cost_accessor_registry.h"
#include "machina/core/platform/env.h"
#include "machina/core/platform/test.h"

namespace machina {
namespace {

class TestGcuCostMeasurement : public CostMeasurement {
 public:
  using CostMeasurement::CostMeasurement;

  absl::Duration GetTotalCost() override { return absl::ZeroDuration(); }
  absl::string_view GetCostType() const override { return "test_gcu"; }
};
REGISTER_COST_MEASUREMENT("test_gcu", TestGcuCostMeasurement);

class TestTpuCostMeasurement : public CostMeasurement {
 public:
  using CostMeasurement::CostMeasurement;

  absl::Duration GetTotalCost() override { return absl::ZeroDuration(); }
  absl::string_view GetCostType() const override { return "test_tpu"; }
};
REGISTER_COST_MEASUREMENT("test_tpu", TestTpuCostMeasurement);

class TestRequestCostAccessor : public RequestCostAccessor {
 public:
  RequestCost* GetRequestCost() const override { return nullptr; }
};
REGISTER_REQUEST_COST_ACCESSOR("test", TestRequestCostAccessor);

TEST(CreateCostMeasurementsTest, Basic) {
  setenv("TF_COST_MEASUREMENT_TYPE", "test_gcu, test_tpu, test_invalid",
         /*overwrite=*/1);
  const CostMeasurement::Context context;
  std::vector<std::unique_ptr<CostMeasurement>> measurements =
      CreateCostMeasurements(context);

  EXPECT_EQ(measurements.size(), 2);
  EXPECT_EQ(measurements[0]->GetTotalCost(), absl::ZeroDuration());
  EXPECT_EQ(measurements[0]->GetCostType(), "test_gcu");
  EXPECT_EQ(measurements[1]->GetTotalCost(), absl::ZeroDuration());
  EXPECT_EQ(measurements[1]->GetCostType(), "test_tpu");
}

TEST(CreateRequestCostAccessorTest, Basic) {
  setenv("TF_REQUEST_COST_ACCESSOR_TYPE", "test", /*overwrite=*/1);
  std::unique_ptr<RequestCostAccessor> test_req_cost_accessor =
      CreateRequestCostAccessor();

  ASSERT_NE(test_req_cost_accessor, nullptr);
  EXPECT_EQ(test_req_cost_accessor->GetRequestCost(), nullptr);
}

}  // namespace
}  // namespace machina
