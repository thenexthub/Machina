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

#include "machina/c/experimental/saved_model/public/saved_model_api.h"

#include <memory>
#include <string>
#include <unordered_set>
#include <utility>

#include "absl/status/status.h"
#include "absl/types/optional.h"
#include "machina/c/eager/tfe_context_internal.h"
#include "machina/c/experimental/saved_model/core/saved_model_api.h"
#include "machina/c/experimental/saved_model/core/tf_saved_model_api.h"
#include "machina/c/experimental/saved_model/internal/concrete_function_list_type.h"
#include "machina/c/experimental/saved_model/internal/concrete_function_type.h"
#include "machina/c/experimental/saved_model/internal/saved_model_api_type.h"
#include "machina/c/experimental/saved_model/internal/signature_def_function_type.h"
#include "machina/c/tf_status.h"
#include "machina/c/tf_status_internal.h"
#include "machina/core/common_runtime/eager/context.h"
#include "machina/core/platform/casts.h"
#include "machina/core/platform/errors.h"
#include "machina/core/platform/status.h"

extern "C" {

TF_SavedModel* TF_LoadSavedModel(const char* dirname, TFE_Context* ctx,
                                 TF_Status* status) {
  std::string saved_model_dir(dirname);
  std::unique_ptr<machina::SavedModelAPI> result;

  if (machina::unwrap(ctx)->UsesTFRT()) {
    status->status = machina::errors::Unimplemented(
        "TFRT SavedModel implementation will be added in the future");
  } else {
    std::unique_ptr<machina::TFSavedModelAPI> saved_model;
    status->status = machina::TFSavedModelAPI::Load(
        dirname, absl::nullopt,
        machina::down_cast<machina::EagerContext*>(
            machina::unwrap(ctx)),
        &saved_model);
    result = std::move(saved_model);
  }

  if (!status->status.ok()) {
    return nullptr;
  }
  return machina::wrap(result.release());
}

TF_SavedModel* TF_LoadSavedModelWithTags(const char* dirname, TFE_Context* ctx,
                                         const char* const* tags, int tags_len,
                                         TF_Status* status) {
  std::string saved_model_dir(dirname);

  std::unordered_set<std::string> tagset;
  for (int i = 0; i < tags_len; ++i) {
    tagset.insert(std::string(tags[i]));
  }

  std::unique_ptr<machina::SavedModelAPI> result;
  if (machina::unwrap(ctx)->UsesTFRT()) {
    status->status = machina::errors::Unimplemented(
        "TFRT SavedModel implementation will be added in the future");
  } else {
    std::unique_ptr<machina::TFSavedModelAPI> saved_model;
    status->status = machina::TFSavedModelAPI::Load(
        dirname, tagset,
        machina::down_cast<machina::EagerContext*>(
            machina::unwrap(ctx)),
        &saved_model);
    result = std::move(saved_model);
  }

  if (!status->status.ok()) {
    return nullptr;
  }
  return machina::wrap(result.release());
}

void TF_DeleteSavedModel(TF_SavedModel* model) {
  delete machina::unwrap(model);
}

TF_ConcreteFunction* TF_GetSavedModelConcreteFunction(TF_SavedModel* model,
                                                      const char* function_path,
                                                      TF_Status* status) {
  machina::ConcreteFunction* result = nullptr;
  absl::Status get_function_status =
      machina::unwrap(model)->GetFunction(function_path, &result);
  status->status.Update(get_function_status);
  if (!get_function_status.ok()) {
    return nullptr;
  }
  return machina::wrap(result);
}

TF_CAPI_EXPORT extern TF_SignatureDefFunction*
TF_GetSavedModelSignatureDefFunction(TF_SavedModel* model,
                                     const char* signature_def_key,
                                     TF_Status* status) {
  machina::SignatureDefFunction* result = nullptr;
  absl::Status get_function_status =
      machina::unwrap(model)->GetSignatureDefFunction(signature_def_key,
                                                         &result);
  status->status.Update(get_function_status);
  if (!get_function_status.ok()) {
    return nullptr;
  }
  return machina::wrap(result);
}

}  // end extern "C"
