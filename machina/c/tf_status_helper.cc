/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 26, 2025.
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

#include "machina/c/tf_status_helper.h"

#include <string>

#include "machina/c/tf_status.h"
#include "machina/xla/tsl/c/tsl_status_helper.h"

namespace tsl {

void Set_TF_Status_from_Status(TF_Status* tf_status,
                               const absl::Status& status) {
  TF_SetStatus(tf_status, TSLCodeFromStatusCode(status.code()),
               absl::StatusMessageAsCStr(status));
  status.ForEachPayload(
      [tf_status](absl::string_view key, const absl::Cord& value) {
        std::string key_str(key);
        std::string value_str(value);
        TF_SetPayload(tf_status, key_str.c_str(), value_str.c_str());
      });
}

absl::Status StatusFromTF_Status(const TF_Status* tf_status) {
  absl::Status status(StatusCodeFromTSLCode(TF_GetCode(tf_status)),
                      TF_Message(tf_status));
  TF_ForEachPayload(
      tf_status,
      [](const char* key, const char* value, void* capture) {
        absl::Status* status = static_cast<absl::Status*>(capture);
        status->SetPayload(key, absl::Cord(absl::string_view(value)));
      },
      &status);
  return status;
}

}  // namespace tsl
