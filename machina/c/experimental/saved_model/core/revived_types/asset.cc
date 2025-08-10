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

#include "machina/c/experimental/saved_model/core/revived_types/asset.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "machina/c/eager/immediate_execution_context.h"
#include "machina/c/eager/immediate_execution_tensor_handle.h"
#include "machina/c/experimental/saved_model/core/revived_types/tensorhandle_convertible.h"
#include "machina/c/tensor_interface.h"
#include "machina/cc/saved_model/constants.h"
#include "machina/core/platform/path.h"
#include "machina/core/platform/status.h"

namespace machina {

Asset::Asset(ImmediateTensorHandlePtr handle)
    : TensorHandleConvertible(std::move(handle)) {}

absl::Status Asset::Create(ImmediateExecutionContext* ctx,
                           const std::string& saved_model_dir,
                           const std::string& asset_filename,
                           std::unique_ptr<Asset>* output) {
  std::string abs_path =
      io::JoinPath(saved_model_dir, kSavedModelAssetsDirectory, asset_filename);
  AbstractTensorPtr tensor(ctx->CreateStringScalar(abs_path));
  if (tensor.get() == nullptr) {
    return absl::InternalError(absl::StrCat(
        "Failed to create scalar string tensor for Asset at path ", abs_path));
  }

  ImmediateTensorHandlePtr handle(ctx->CreateLocalHandle(tensor.get()));
  output->reset(new Asset(std::move(handle)));
  return absl::Status();
}

}  // namespace machina
