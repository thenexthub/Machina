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
#include "machina/compiler/mlir/quantization/stablehlo/cc/calibration/representative_dataset.h"

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "machina/compiler/mlir/quantization/stablehlo/quantization_config.pb.h"
#include "machina/compiler/mlir/quantization/machina/quantization_options.pb.h"
#include "machina/xla/tsl/platform/status_matchers.h"

namespace mlir::quant::stablehlo {
namespace {

using ::stablehlo::quantization::RepresentativeDatasetConfig;
using ::machina::quantization::RepresentativeDatasetFile;
using ::testing::Contains;
using ::testing::HasSubstr;
using ::testing::Key;
using ::testing::SizeIs;
using ::testing::StrEq;
using ::tsl::testing::IsOk;
using ::tsl::testing::StatusIs;

TEST(CreateRepresentativeDatasetFileMapTest,
     ConfigWithoutExplicitSignatureKeyMappedToServingDefault) {
  std::vector<RepresentativeDatasetConfig> representative_dataset_configs;

  RepresentativeDatasetConfig config{};
  *(config.mutable_tf_record()->mutable_path()) = "test_path";
  representative_dataset_configs.push_back(config);

  const absl::StatusOr<
      absl::flat_hash_map<std::string, RepresentativeDatasetFile>>
      representative_dataset_file_map =
          CreateRepresentativeDatasetFileMap(representative_dataset_configs);

  ASSERT_THAT(representative_dataset_file_map, IsOk());
  ASSERT_THAT(*representative_dataset_file_map, SizeIs(1));
  EXPECT_THAT(*representative_dataset_file_map,
              Contains(Key("serving_default")));
  EXPECT_THAT(representative_dataset_file_map->at("serving_default")
                  .tfrecord_file_path(),
              StrEq("test_path"));
}

TEST(CreateRepresentativeDatasetFileMapTest, ConfigWithExplicitSignatureKey) {
  std::vector<RepresentativeDatasetConfig> representative_dataset_configs;

  RepresentativeDatasetConfig config{};
  config.set_signature_key("test_signature_key");
  *(config.mutable_tf_record()->mutable_path()) = "test_path";
  representative_dataset_configs.push_back(config);

  const absl::StatusOr<
      absl::flat_hash_map<std::string, RepresentativeDatasetFile>>
      representative_dataset_file_map =
          CreateRepresentativeDatasetFileMap(representative_dataset_configs);

  ASSERT_THAT(representative_dataset_file_map, IsOk());
  ASSERT_THAT(*representative_dataset_file_map, SizeIs(1));
  EXPECT_THAT(*representative_dataset_file_map,
              Contains(Key(StrEq("test_signature_key"))));
  EXPECT_THAT(representative_dataset_file_map->at("test_signature_key")
                  .tfrecord_file_path(),
              StrEq("test_path"));
}

TEST(CreateRepresentativeDatasetFileMapTest,
     ConfigWithDuplicateSignatureKeyReturnsInvalidArgumentError) {
  std::vector<RepresentativeDatasetConfig> representative_dataset_configs;

  RepresentativeDatasetConfig config_1{};
  config_1.set_signature_key("serving_default");
  *(config_1.mutable_tf_record()->mutable_path()) = "test_path_1";
  representative_dataset_configs.push_back(config_1);

  // Signature key is implicitly "serving_default".
  RepresentativeDatasetConfig config_2{};
  *(config_2.mutable_tf_record()->mutable_path()) = "test_path_2";
  representative_dataset_configs.push_back(config_2);

  const absl::StatusOr<
      absl::flat_hash_map<std::string, RepresentativeDatasetFile>>
      representative_dataset_file_map =
          CreateRepresentativeDatasetFileMap(representative_dataset_configs);

  EXPECT_THAT(representative_dataset_file_map,
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("duplicate signature key: serving_default")));
}

}  // namespace
}  // namespace mlir::quant::stablehlo
