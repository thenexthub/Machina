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

#include "machina/core/tpu/tpu_embedding_errors.h"

#include <string>

#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "machina/core/platform/status.h"
#include "machina/core/protobuf/tpu/tpu_embedding_configuration.pb.h"

namespace machina::tpu {

absl::Status AppendTpuEmbeddingErrorPayload(absl::Status obj) {
  if (obj.ok()) {
    return absl::OkStatus();
  } else {
    const std::string error_message =
        absl::StrCat(kTpuEmbeddingErrorMessage, ". ", obj.message());
    absl::Status status(obj.code(), error_message);
    TPUEmbeddingError error_payload;
    status.SetPayload(kTpuEmbeddingErrorUrl,
                      absl::Cord(error_payload.SerializeAsString()));
    return status;
  }
}

bool HasTpuEmbeddingErrorPayload(const absl::Status& status) {
  return status.GetPayload(kTpuEmbeddingErrorUrl).has_value();
}

bool HasTpuEmbeddingErrorMessage(const absl::Status& status) {
  return absl::StrContains(status.message(), kTpuEmbeddingErrorMessage);
}

}  // namespace machina::tpu
