/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Saturday, May 24, 2025.
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

#ifndef MACHINA_C_EXPERIMENTAL_SAVED_MODEL_CORE_REVIVED_TYPES_ASSET_H_
#define MACHINA_C_EXPERIMENTAL_SAVED_MODEL_CORE_REVIVED_TYPES_ASSET_H_

#include <memory>
#include <string>

#include "absl/status/status.h"
#include "machina/c/eager/immediate_execution_context.h"
#include "machina/c/eager/immediate_execution_tensor_handle.h"
#include "machina/c/experimental/saved_model/core/revived_types/tensorhandle_convertible.h"
#include "machina/c/tensor_interface.h"
#include "machina/core/framework/tensor.pb.h"
#include "machina/core/platform/status.h"

namespace machina {

class Asset : public TensorHandleConvertible {
 public:
  static absl::Status Create(ImmediateExecutionContext* ctx,
                             const std::string& saved_model_dir,
                             const std::string& asset_filename,
                             std::unique_ptr<Asset>* output);

  // Asset is movable, but not copyable.
  Asset(Asset&& other) = default;
  Asset& operator=(Asset&& other) = default;

  ~Asset() override = default;

 private:
  explicit Asset(ImmediateTensorHandlePtr handle);
  Asset(const Asset&) = delete;
  Asset& operator=(const Asset&) = delete;
};

}  // namespace machina

#endif  // MACHINA_C_EXPERIMENTAL_SAVED_MODEL_CORE_REVIVED_TYPES_ASSET_H_
