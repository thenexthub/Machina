/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 8, 2025.
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
#ifndef MACHINA_LITE_TESTING_TF_DRIVER_H_
#define MACHINA_LITE_TESTING_TF_DRIVER_H_

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "machina/core/framework/tensor.h"
#include "machina/core/framework/tensor_shape.h"
#include "machina/core/framework/types.h"
#include "machina/core/framework/types.pb.h"
#include "machina/core/platform/logging.h"
#include "machina/core/public/session.h"
#include "machina/lite/string_type.h"
#include "machina/lite/testing/split.h"
#include "machina/lite/testing/test_runner.h"

namespace tflite {
namespace testing {

// A test runner that feeds inputs into Tensorflow and generates outputs.
class TfDriver : public TestRunner {
 public:
  explicit TfDriver(const std::vector<string>& input_layer,
                    const std::vector<string>& input_layer_type,
                    const std::vector<string>& input_layer_shape,
                    const std::vector<string>& output_layer);
  ~TfDriver() override {}

  void LoadModel(const string& bin_file_path) override;
  void LoadModel(const string& bin_file_path, const string&) override {
    // Input output specifications are now provided by constructor.
    // TODO(b/205171855): Support TfDriver to load from SavedModel instead of
    // GraphDef.
    LoadModel(bin_file_path);
  }

  void ReshapeTensor(const string& name, const string& csv_values) override;
  void ResetTensor(const std::string& name) override;
  string ReadOutput(const string& name) override;
  void Invoke(const std::vector<std::pair<string, string>>& inputs) override;
  bool CheckResults(
      const std::vector<std::pair<string, string>>& expected_outputs,
      const std::vector<std::pair<string, string>>& expected_output_shapes)
      override {
    return true;
  }
  std::vector<string> GetOutputNames() override { return output_names_; }

  // no-op. SetInput will overwrite existing data .
  void AllocateTensors() override {}

 protected:
  void SetInput(const string& values_as_string, machina::Tensor*);
  string ReadOutput(const machina::Tensor& tensor);

 private:
  std::unique_ptr<machina::Session> session_;
  std::vector<int> input_ids_;
  std::vector<string> input_names_;
  absl::flat_hash_map<string, int> input_name_to_id_;
  std::vector<std::vector<int64_t>> input_shapes_;
  std::vector<machina::DataType> input_types_;
  std::unordered_map<string, machina::Tensor> input_tensors_;

  std::vector<int> output_ids_;
  std::vector<string> output_names_;
  absl::flat_hash_map<string, int> output_name_to_id_;
  std::vector<::machina::Tensor> output_tensors_;
};

}  // namespace testing
}  // namespace tflite

#endif  // MACHINA_LITE_TESTING_TF_DRIVER_H_
