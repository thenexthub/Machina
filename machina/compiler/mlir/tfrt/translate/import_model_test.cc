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

#include "machina/compiler/mlir/tfrt/translate/import_model.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "machina/compiler/mlir/tfrt/translate/tfrt_compile_options.h"

namespace machina {
namespace {

using ::testing::SizeIs;

TEST(GetTfrtPipelineOptions, BatchPaddingPolicy) {
  machina::TfrtCompileOptions options;
  options.batch_padding_policy = "PAD_TEST_OPTION";
  auto pipeline_options = GetTfrtPipelineOptions(options);
  EXPECT_EQ(pipeline_options->batch_padding_policy, "PAD_TEST_OPTION");
}

TEST(GetTfrtPipelineOptions, NumBatchThreads) {
  machina::TfrtCompileOptions options;
  options.batch_options.set_num_batch_threads(2);
  auto pipeline_options = GetTfrtPipelineOptions(options);
  EXPECT_EQ(pipeline_options->num_batch_threads, 2);
}

TEST(GetTfrtPipelineOptions, MaxBatchSize) {
  machina::TfrtCompileOptions options;
  options.batch_options.set_max_batch_size(8);
  auto pipeline_options = GetTfrtPipelineOptions(options);
  EXPECT_EQ(pipeline_options->max_batch_size, 8);
}

TEST(GetTfrtPipelineOptions, BatchTimeoutMicros) {
  machina::TfrtCompileOptions options;
  options.batch_options.set_batch_timeout_micros(5000);
  auto pipeline_options = GetTfrtPipelineOptions(options);
  EXPECT_EQ(pipeline_options->batch_timeout_micros, 5000);
}

TEST(GetTfrtPipelineOptions, AllowedBatchSizes) {
  machina::TfrtCompileOptions options;
  options.batch_options.add_allowed_batch_sizes(2);
  options.batch_options.add_allowed_batch_sizes(4);
  options.batch_options.add_allowed_batch_sizes(8);
  auto pipeline_options = GetTfrtPipelineOptions(options);
  EXPECT_THAT(pipeline_options->allowed_batch_sizes, SizeIs(3));
}

TEST(GetTfrtPipelineOptions, MaxEnqueuedBatches) {
  machina::TfrtCompileOptions options;
  options.batch_options.set_max_enqueued_batches(250);
  auto pipeline_options = GetTfrtPipelineOptions(options);
  EXPECT_EQ(pipeline_options->max_enqueued_batches, 250);
}

}  // namespace
}  // namespace machina
