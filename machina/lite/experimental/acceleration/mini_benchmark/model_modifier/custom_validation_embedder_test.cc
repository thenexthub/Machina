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
#include "machina/lite/experimental/acceleration/mini_benchmark/model_modifier/custom_validation_embedder.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "machina/lite/core/c/c_api_types.h"
#include "machina/lite/core/interpreter.h"
#include "machina/lite/core/interpreter_builder.h"
#include "machina/lite/core/kernels/register.h"
#include "machina/lite/core/model_builder.h"
#include "machina/lite/core/subgraph.h"
#include "machina/lite/experimental/acceleration/mini_benchmark/call_register.h"
#include "machina/lite/experimental/acceleration/mini_benchmark/embedded_mobilenet_model.h"
#include "machina/lite/experimental/acceleration/mini_benchmark/mini_benchmark_test_helper.h"
#include "machina/lite/experimental/acceleration/mini_benchmark/status_codes.h"
#include "machina/lite/kernels/internal/tensor_ctypes.h"
#include "machina/lite/schema/schema_generated.h"
#include "machina/lite/tools/model_loader.h"

namespace tflite {
namespace acceleration {
namespace {
using ::flatbuffers::FlatBufferBuilder;

constexpr int kMobileNetModelInputByteSize = 1 * 224 * 224 * 3;

class CustomValidationEmbedderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    std::string plain_model_path = MiniBenchmarkTestHelper::DumpToTempFile(
        "mobilenet_quant.tflite",
        g_tflite_acceleration_embedded_mobilenet_model,
        g_tflite_acceleration_embedded_mobilenet_model_len);
    ASSERT_TRUE(!plain_model_path.empty());
    plain_model_loader_ =
        std::make_unique<tools::PathModelLoader>(plain_model_path);
    ASSERT_TRUE(plain_model_loader_->Init());
  }

  std::unique_ptr<tools::ModelLoader> plain_model_loader_;
};

TEST_F(CustomValidationEmbedderTest, BuildValidationModelSucceed) {
  int batch_size = 5;
  std::vector<uint8_t> input_buffer(batch_size * kMobileNetModelInputByteSize);
  CustomValidationEmbedder embedder(batch_size, {input_buffer});

  FlatBufferBuilder fbb;
  EXPECT_EQ(
      embedder.BuildModel(*plain_model_loader_->GetModel()->GetModel(), fbb),
      kMinibenchmarkSuccess);

  // Verify validation graph can be invoked.
  auto model =
      FlatBufferModel::BuildFromModel(GetModel(fbb.GetBufferPointer()));
  auto interpreter = std::make_unique<Interpreter>();
  auto resolver = std::make_unique<
      ::tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates>();
  resolver->AddCustom("validation/call", ops::Register_CALL(), 1);
  ASSERT_EQ(InterpreterBuilder(*model, *resolver)(&interpreter), kTfLiteOk);
  ASSERT_NE(interpreter, nullptr);
  Subgraph* validation_graph = interpreter->subgraph(1);
  EXPECT_THAT(input_buffer, testing::ElementsAreArray(
                                GetTensorData<uint8_t>(validation_graph->tensor(
                                    validation_graph->inputs()[0])),
                                input_buffer.size()));
  EXPECT_EQ(validation_graph->AllocateTensors(), kTfLiteOk);
  EXPECT_EQ(validation_graph->Invoke(), kTfLiteOk);
}

TEST_F(CustomValidationEmbedderTest, BuildValidationModelTooManyInput) {
  int batch_size = 5;
  CustomValidationEmbedder embedder(batch_size, {{}, {}});

  FlatBufferBuilder fbb;
  EXPECT_EQ(
      embedder.BuildModel(*plain_model_loader_->GetModel()->GetModel(), fbb),
      kMinibenchmarkValidationSubgraphBuildFailed);
}

TEST_F(CustomValidationEmbedderTest, BuildValidationModelInvalidBufferSize) {
  CustomValidationEmbedder embedder(2, {std::vector<uint8_t>(2, 2)});

  FlatBufferBuilder fbb;
  EXPECT_EQ(
      embedder.BuildModel(*plain_model_loader_->GetModel()->GetModel(), fbb),
      kMinibenchmarkValidationSubgraphBuildFailed);
}

}  // namespace
}  // namespace acceleration
}  // namespace tflite
