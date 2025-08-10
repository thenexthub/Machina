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
#include "machina/core/data/service/validate_utils.h"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

#include "absl/random/random.h"
#include "absl/strings/str_cat.h"
#include "machina/core/framework/dataset.h"
#include "machina/core/framework/tensor_shape.pb.h"
#include "machina/core/lib/core/status_test_util.h"
#include "machina/core/platform/status_matchers.h"
#include "machina/core/platform/test.h"
#include "machina/core/protobuf/data_service.pb.h"
#include "machina/core/protobuf/error_codes.pb.h"
#include "machina/core/protobuf/struct.pb.h"

namespace machina {
namespace data {
namespace {

using ::machina::testing::StatusIs;
using ::testing::AllOf;
using ::testing::HasSubstr;

DataServiceMetadata GetDefaultMetadata() {
  // Fills the Tensor shape with a shuffled range of 1, 2, ..., 10.
  static absl::BitGen bitgen;
  std::vector<int64_t> dims(10);
  std::iota(dims.begin(), dims.end(), 1);
  std::shuffle(dims.begin(), dims.end(), bitgen);
  StructuredValue decoded_spec;
  for (size_t i = 0; i < dims.size(); ++i) {
    TensorShapeProto::Dim* dim =
        decoded_spec.mutable_tensor_shape_value()->add_dim();
    dim->set_size(dims[i]);
    dim->set_name(absl::StrCat("dim", dims[i]));
  }

  DataServiceMetadata metadata;
  metadata.set_element_spec(decoded_spec.SerializeAsString());
  metadata.set_compression(DataServiceMetadata::COMPRESSION_SNAPPY);
  metadata.set_cardinality(kInfiniteCardinality);
  return metadata;
}

TEST(ValidateUtilsTest, ValidateMatchingDataset) {
  DataServiceMetadata metadata1 = GetDefaultMetadata();
  DataServiceMetadata metadata2 = GetDefaultMetadata();
  TF_EXPECT_OK(ValidateMatchingDataset("dataset_id", metadata1, metadata2));
}

TEST(ValidateUtilsTest, EmptyMetadata) {
  DataServiceMetadata metadata1;
  DataServiceMetadata metadata2;
  TF_EXPECT_OK(ValidateMatchingDataset("dataset_id", metadata1, metadata2));
}

TEST(ValidateUtilsTest, ElementSpecDoesNotMatch) {
  DataServiceMetadata metadata1 = GetDefaultMetadata();
  DataServiceMetadata metadata2 = GetDefaultMetadata();
  metadata2.clear_element_spec();
  EXPECT_THAT(ValidateMatchingDataset("dataset_id", metadata1, metadata2),
              absl_testing::StatusIs(
                  error::INVALID_ARGUMENT,
                  HasSubstr("got diff for dataset ID dataset_id with different "
                            "element_spec")));

  StructuredValue different_decoded_spec;
  different_decoded_spec.mutable_tensor_shape_value()->add_dim()->set_size(1);
  different_decoded_spec.mutable_tensor_shape_value()->add_dim()->set_size(2);
  metadata2.set_element_spec(different_decoded_spec.SerializeAsString());
  EXPECT_THAT(ValidateMatchingDataset("dataset_id", metadata1, metadata2),
              absl_testing::StatusIs(
                  error::INVALID_ARGUMENT,
                  HasSubstr("got diff for dataset ID dataset_id with different "
                            "element_spec")));
}

TEST(ValidateUtilsTest, InvalidElementSpec) {
  DataServiceMetadata metadata1 = GetDefaultMetadata();
  DataServiceMetadata metadata2 = GetDefaultMetadata();
  metadata1.set_element_spec("Invalid element spec");

  EXPECT_THAT(ValidateMatchingDataset("dataset_id", metadata1, metadata2),
              absl_testing::StatusIs(
                  error::INVALID_ARGUMENT,
                  "Failed to parse element_spec for dataset dataset_id: "
                  "Invalid element spec."));
}

TEST(ValidateUtilsTest, CompressionDoesNotMatch) {
  DataServiceMetadata metadata1 = GetDefaultMetadata();
  DataServiceMetadata metadata2 = GetDefaultMetadata();
  metadata2.set_compression(DataServiceMetadata::COMPRESSION_OFF);

  EXPECT_THAT(ValidateMatchingDataset("dataset_id", metadata1, metadata2),
              absl_testing::StatusIs(
                  error::INVALID_ARGUMENT,
                  AllOf(HasSubstr("got diff for dataset ID dataset_id"),
                        HasSubstr("modified: compression"))));
}

TEST(ValidateUtilsTest, CardinalityDoesNotMatch) {
  DataServiceMetadata metadata1 = GetDefaultMetadata();
  DataServiceMetadata metadata2 = GetDefaultMetadata();
  metadata2.set_cardinality(kUnknownCardinality);

  EXPECT_THAT(ValidateMatchingDataset("dataset_id", metadata1, metadata2),
              absl_testing::StatusIs(
                  error::INVALID_ARGUMENT,
                  AllOf(HasSubstr("got diff for dataset ID dataset_id"),
                        HasSubstr("modified: cardinality"))));
}

}  // namespace
}  // namespace data
}  // namespace machina
