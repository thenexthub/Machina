/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 26, 2025.
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

#include "machina/core/lib/monitoring/metric_def.h"

#include "machina/core/platform/test.h"

namespace machina {
namespace monitoring {
namespace {

TEST(MetricDefTest, Simple) {
  const MetricDef<MetricKind::kCumulative, int64_t, 0> metric_def0(
      "/machina/metric0", "An example metric with no labels.");
  const MetricDef<MetricKind::kGauge, HistogramProto, 1> metric_def1(
      "/machina/metric1", "An example metric with one label.", "LabelName");

  EXPECT_EQ("/machina/metric0", metric_def0.name());
  EXPECT_EQ("/machina/metric1", metric_def1.name());

  EXPECT_EQ(MetricKind::kCumulative, metric_def0.kind());
  EXPECT_EQ(MetricKind::kGauge, metric_def1.kind());

  EXPECT_EQ("An example metric with no labels.", metric_def0.description());
  EXPECT_EQ("An example metric with one label.", metric_def1.description());

  EXPECT_EQ(0, metric_def0.label_descriptions().size());
  ASSERT_EQ(1, metric_def1.label_descriptions().size());
  EXPECT_EQ("LabelName", metric_def1.label_descriptions()[0]);
}

TEST(MetricDefTest, StringsPersist) {
  // Ensure string attributes of the metric are copied into the metric
  string name = "/machina/metric0";
  string description = "test description";
  string label_description = "test label description";
  const MetricDef<MetricKind::kCumulative, int64_t, 1> metric_def(
      name, description, label_description);

  // Mutate the strings
  name[4] = 'A';
  description[4] = 'B';
  label_description[4] = 'C';

  EXPECT_NE(name, metric_def.name());
  EXPECT_NE(description, metric_def.description());
  EXPECT_NE(label_description, metric_def.label_descriptions()[0]);
}

}  // namespace
}  // namespace monitoring
}  // namespace machina
