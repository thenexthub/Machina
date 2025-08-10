/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 15, 2025.
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

#include "machina/compiler/mlir/machina/utils/xla_sharding_util.h"

#include <string>

#include <gtest/gtest.h>
#include "toolchain/ADT/StringRef.h"
#include "mlir/IR/Attributes.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "machina/xla/xla_data.pb.h"

inline constexpr toolchain::StringRef kXlaShardingAttrName = "_XlaSharding";

namespace machina {
namespace {

TEST(DecodeShardingAttributeTest, CheckInvalidString) {
  xla::OpSharding sharding;
  EXPECT_TRUE(DecodeShardingAttribute("", sharding).succeeded());
  EXPECT_TRUE(DecodeShardingAttribute("manual", sharding).failed());
}

TEST(DecodeShardingAttributeTest, CheckManualShardString) {
  xla::OpSharding sharding;
  EXPECT_TRUE(DecodeShardingAttribute("{manual}", sharding).succeeded());
  EXPECT_TRUE(sharding.type() == sharding.MANUAL);
  EXPECT_EQ(0, sharding.tile_assignment_devices_size());
}

TEST(DecodeShardingAttributeTest, CheckMaximalShardString) {
  xla::OpSharding sharding;
  EXPECT_TRUE(
      DecodeShardingAttribute("{maximal device=0}", sharding).succeeded());
  EXPECT_TRUE(sharding.type() == sharding.MAXIMAL);
  EXPECT_EQ(1, sharding.tile_assignment_devices_size());
}

TEST(ShardingEquivalenceTest, CheckOpShardingEquivalent) {
  xla::OpSharding sharding1;
  EXPECT_TRUE(
      DecodeShardingAttribute("{devices=[2,1]0,1}", sharding1).succeeded());
  xla::OpSharding sharding2;
  EXPECT_TRUE(
      DecodeShardingAttribute("{devices=[2,1]<=[2]}", sharding2).succeeded());
  EXPECT_TRUE(VerifyShardingEquivalent(sharding1, sharding2).succeeded());
}

TEST(ShardingEquivalenceTest, CheckOpShardingNotEquivalent) {
  xla::OpSharding sharding1;
  EXPECT_TRUE(
      DecodeShardingAttribute("{devices=[2,1]1,0}", sharding1).succeeded());
  xla::OpSharding sharding2;
  EXPECT_TRUE(
      DecodeShardingAttribute("{devices=[2,1]<=[2]}", sharding2).succeeded());
  EXPECT_FALSE(VerifyShardingEquivalent(sharding1, sharding2).succeeded());
}
}  // namespace

}  // namespace machina
