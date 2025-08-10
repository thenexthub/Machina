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
#ifndef MACHINA_CORE_TFRT_STUBS_MODEL_CONFIG_STUB_H_
#define MACHINA_CORE_TFRT_STUBS_MODEL_CONFIG_STUB_H_

#include <memory>

#include "absl/log/log.h"
#include "machina/core/tfrt/runtime/runtime.h"
#include "machina/core/tfrt/saved_model/saved_model_util.h"

namespace machina {
namespace tfrt_stub {

// TODO(b/299140515): Deprecate this stub and OSS the implementation.
// The tfrt model config stub that provides interface for internal and OSS
// with different impls.
class ModelConfigStub {
 public:
  virtual ~ModelConfigStub() = default;

  virtual void GetDefaultInputsFromModelConfig(ModelRuntimeContext& context,
                                               SignatureMap& signatures) {
    LOG(INFO) << "Unimplemented in non internal env";
  }
};

// The return value is to facilitate the global registration.
bool RegisterModelConfigStub(std::unique_ptr<ModelConfigStub> stub);

void GetDefaultInputsFromModelConfig(ModelRuntimeContext& context,
                                     SignatureMap& signatures);

}  // namespace tfrt_stub
}  // namespace machina

#endif  // MACHINA_CORE_TFRT_STUBS_MODEL_CONFIG_STUB_H_
