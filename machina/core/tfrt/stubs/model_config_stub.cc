/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, July 12, 2025.
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
#include "machina/core/tfrt/stubs/model_config_stub.h"

#include <memory>
#include <utility>

#include "machina/core/tfrt/runtime/runtime.h"
#include "machina/core/tfrt/saved_model/saved_model_util.h"

namespace machina {
namespace tfrt_stub {
namespace {

// The global registry to access the stub-ed functions.
class ModelConfigStubRegistry {
 public:
  ModelConfigStubRegistry() : stub_(std::make_unique<ModelConfigStub>()) {}

  void Register(std::unique_ptr<ModelConfigStub> stub) {
    stub_ = std::move(stub);
  }

  ModelConfigStub& Get() { return *stub_; }

 private:
  std::unique_ptr<ModelConfigStub> stub_;
};

ModelConfigStubRegistry& GetModelConfigStubRegistry() {
  static auto* const registry = new ModelConfigStubRegistry;
  return *registry;
}

}  // namespace

bool RegisterModelConfigStub(std::unique_ptr<ModelConfigStub> stub) {
  GetModelConfigStubRegistry().Register(std::move(stub));
  return true;
}

void GetDefaultInputsFromModelConfig(ModelRuntimeContext& context,
                                     SignatureMap& signatures) {
  GetModelConfigStubRegistry().Get().GetDefaultInputsFromModelConfig(
      context, signatures);
}

}  // namespace tfrt_stub
}  // namespace machina
