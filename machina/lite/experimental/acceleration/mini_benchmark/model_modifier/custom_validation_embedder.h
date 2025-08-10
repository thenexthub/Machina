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
#ifndef MACHINA_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_MODEL_MODIFIER_CUSTOM_VALIDATION_EMBEDDER_H_
#define MACHINA_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_MODEL_MODIFIER_CUSTOM_VALIDATION_EMBEDDER_H_

#include <utility>
#include <vector>

#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "machina/lite/core/api/error_reporter.h"
#include "machina/lite/experimental/acceleration/mini_benchmark/status_codes.h"
#include "machina/lite/schema/schema_generated.h"
#include "machina/lite/stderr_reporter.h"

namespace tflite {
namespace acceleration {

// Create a model with custom validation graph.
//
// 'validation model' (new subgraph)
// input (batch_size)
//           |
// +-----------------------+
// |'main_model' (0)       |
// | +---------------+     |
// | |input          +---+ |
// | +---------------+   | |
// |                     ~ |
// | +---------------+   | |
// | |outputs        +<--+ |
// | +---------------+     |
// |                       |
// +-----------------------+
//           |
// output (batch_size)
//
// The new model contains all the information from main_model, with an extra
// subgraph for validation purposes. The validation graph calls the primary
// subgraph with batch_size. The input data is embedded to the validation graph.
// custom_input should have the same order as the input in the main_model. E.g.
// custom_input[i] will be mapped to main_model.input[i].
class CustomValidationEmbedder {
 public:
  CustomValidationEmbedder(
      int batch_size, std::vector<std::vector<uint8_t>> custom_input,
      ErrorReporter* error_reporter = DefaultErrorReporter())
      : batch_size_(batch_size),
        custom_input_(std::move(custom_input)),
        error_reporter_(error_reporter) {}

  // Move only.
  CustomValidationEmbedder(CustomValidationEmbedder&&) = default;
  CustomValidationEmbedder& operator=(CustomValidationEmbedder&&) = default;

  // Build the final model with main_model and validation subgraph.
  MinibenchmarkStatus BuildModel(const Model& main_model,
                                 flatbuffers::FlatBufferBuilder& fbb);

 private:
  // Helper function to create tensors in validation graph based on primary
  // subgraph. This function creates new tensors and buffers based on the
  // from_subgraphs.tensors[from_indexes]. The new tensors will have shape[0]
  // set to batch_size_, and indexes stored in new_indexes.
  // New buffers will be created for each of the new tensors, and buffer data is
  // copied from the corresponding buffer_content.
  void CreateTensorsFrom(const SubGraph& from_subgraph,
                         const std::vector<int>& from_indexes,
                         std::vector<std::vector<uint8_t>>* buffer_content,
                         flatbuffers::FlatBufferBuilder& fbb,
                         std::vector<int>& new_indexes,
                         std::vector<flatbuffers::Offset<Buffer>>& buffers,
                         std::vector<flatbuffers::Offset<Tensor>>& tensors);

  int batch_size_;
  std::vector<std::vector<uint8_t>> custom_input_;
  ErrorReporter* error_reporter_;
};

}  // namespace acceleration
}  // namespace tflite

#endif  // MACHINA_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_MODEL_MODIFIER_CUSTOM_VALIDATION_EMBEDDER_H_
