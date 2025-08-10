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
#ifndef MACHINA_CORE_TFRT_SAVED_MODEL_SAVED_MODEL_TESTUTIL_H_
#define MACHINA_CORE_TFRT_SAVED_MODEL_SAVED_MODEL_TESTUTIL_H_

#include <stdlib.h>

#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "machina/cc/saved_model/loader.h"
#include "machina/compiler/mlir/tfrt/translate/tfrt_compile_options.h"
#include "machina/core/tfrt/runtime/runtime.h"
#include "machina/core/tfrt/saved_model/saved_model.h"
#include "tfrt/host_context/host_context.h"  // from @tf_runtime

#if defined(PLATFORM_GOOGLE)
ABSL_DECLARE_FLAG(bool, enable_optimizer);
ABSL_DECLARE_FLAG(std::string, force_data_format);
#endif

namespace machina {
namespace tfrt_stub {

std::unique_ptr<machina::tfrt_stub::Runtime> DefaultTfrtRuntime(
    int num_threads);

struct UserSavedModelOptions {
  bool enable_mlrt = false;
  bool enable_optimizer = false;
  bool enable_grappler = false;
  std::string force_data_format = "";
  machina::SessionMetadata session_metadata;
};

SavedModel::Options DefaultSavedModelOptions(
    machina::tfrt_stub::Runtime* runtime,
    std::optional<UserSavedModelOptions> user_options = std::nullopt);

class TFRTSavedModelTest {
 public:
  explicit TFRTSavedModelTest(const std::string& saved_model_dir);
  TFRTSavedModelTest(const std::string& saved_model_dir,
                     std::unique_ptr<machina::tfrt_stub::Runtime> runtime);

  SavedModel* GetSavedModel() { return saved_model_.get(); }

  tfrt::HostContext* GetHostContext() const {
    return saved_model_->GetHostContext();
  }

 private:
  std::unique_ptr<machina::tfrt_stub::Runtime> runtime_;
  std::unique_ptr<SavedModel> saved_model_;
};

template <typename T, typename U = T>
machina::Tensor CreateTfTensor(absl::Span<const int64_t> shape,
                                  absl::Span<const U> data) {
  machina::Tensor tensor(machina::DataTypeToEnum<T>::value,
                            machina::TensorShape(shape));
  auto flat = tensor.flat<T>();
  for (int i = 0; i < data.size(); ++i) {
    flat(i) = data[i];
  }
  return tensor;
}

template <typename T>
std::vector<T> GetTfTensorData(const machina::Tensor& tensor) {
  return std::vector<T>(tensor.flat<T>().data(),
                        tensor.flat<T>().data() + tensor.NumElements());
}

inline machina::Tensor CreateTfStringTensor(
    absl::Span<const int64_t> shape, absl::Span<const std::string> data) {
  return CreateTfTensor<machina::tstring>(shape, data);
}

void ComputeCurrentTFResult(const std::string& saved_model_dir,
                            const std::string& signature_name,
                            const std::vector<std::string>& input_names,
                            const std::vector<machina::Tensor>& inputs,
                            const std::vector<std::string>& output_names,
                            std::vector<machina::Tensor>* outputs,
                            bool enable_mlir_bridge = false,
                            bool disable_grappler = false);

// Compute the results using TF1 session loaded from the saved model. In
// addition to returning the result tensors, it also fills `bundle` with the
// loaded savedmodel. This is useful as sometimes the result tensors may only be
// valid when the bundle is alive.
void ComputeCurrentTFResult(const std::string& saved_model_dir,
                            const std::string& signature_name,
                            const std::vector<std::string>& input_names,
                            const std::vector<machina::Tensor>& inputs,
                            const std::vector<std::string>& output_names,
                            std::vector<machina::Tensor>* outputs,
                            machina::SavedModelBundle* bundle,
                            bool enable_mlir_bridge = false,
                            bool disable_grappler = false);

void ExpectTensorEqual(const machina::Tensor& x, const machina::Tensor& y,
                       std::optional<double> error = std::nullopt);

SavedModel::Options DefaultTpuModelOptions(
    machina::tfrt_stub::Runtime* runtime,
    machina::TfrtDeviceInfraTarget device_target);

}  // namespace tfrt_stub
}  // namespace machina

#endif  // MACHINA_CORE_TFRT_SAVED_MODEL_SAVED_MODEL_TESTUTIL_H_
