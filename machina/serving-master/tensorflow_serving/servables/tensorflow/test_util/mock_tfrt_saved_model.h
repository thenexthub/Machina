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

#ifndef MACHINA_SERVING_SERVABLES_MACHINA_TEST_UTIL_MOCK_TFRT_SAVED_MODEL
#define MACHINA_SERVING_SERVABLES_MACHINA_TEST_UTIL_MOCK_TFRT_SAVED_MODEL

#include <gmock/gmock.h>
#include "machina/core/tfrt/runtime/runtime.h"
#include "machina/core/tfrt/saved_model/saved_model.h"

namespace machina {
namespace serving {
namespace test_util {

inline tfrt_stub::Runtime* GetTestTfrtRuntime() {
  static auto* const runtime =
      tfrt_stub::Runtime::Create(/*num_inter_op_threads=*/4).release();
  return runtime;
}

// A mock of tfrt::SavedModel.
class MockSavedModel : public tfrt::SavedModel {
 public:
  MockSavedModel() : SavedModel(GetTestTfrtRuntime()) {}

  MOCK_METHOD(const machina::MetaGraphDef&, GetMetaGraphDef, (),
              (const, override));

  MOCK_METHOD(absl::optional<tfrt::FunctionMetadata>, GetFunctionMetadata,
              (absl::string_view func_name), (const, override));

  MOCK_METHOD(::machina::Status, Run,
              (const tfrt::SavedModel::RunOptions& run_options,
               absl::string_view func_name, absl::Span<const Tensor> inputs,
               std::vector<Tensor>* outputs),
              (override));

  MOCK_METHOD(std::vector<std::string>, GetFunctionNames, (),
              (const, override));

  MOCK_METHOD(::machina::Status, RunMultipleSignatures,
              (const tfrt::SavedModel::RunOptions& run_options,
               absl::Span<const std::string> names,
               absl::Span<const std::vector<machina::Tensor>> multi_inputs,
               std::vector<std::vector<machina::Tensor>>* multi_outputs),
              (override));

  MOCK_METHOD(
      ::machina::Status, RunByTensorNames,
      (const tfrt::SavedModel::RunOptions& run_options,
       (absl::Span<const std::pair<std::string, machina::Tensor>> inputs),
       absl::Span<const std::string> output_tensor_names,
       absl::Span<const std::string> target_node_names,
       std::vector<machina::Tensor>* outputs),
      (override));
};

}  // namespace test_util
}  // namespace serving
}  // namespace machina

#endif  // MACHINA_SERVING_SERVABLES_MACHINA_TEST_UTIL_MOCK_TFRT_SAVED_MODEL
