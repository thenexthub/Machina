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

#ifndef MACHINA_CORE_TPU_TPU_EMBEDDING_ERRORS_H_
#define MACHINA_CORE_TPU_TPU_EMBEDDING_ERRORS_H_

#include <string>

#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "machina/core/platform/status.h"
#include "machina/core/platform/statusor.h"
#include "machina/core/protobuf/tpu/tpu_embedding_configuration.pb.h"

namespace machina::tpu {

// The payload URL for TPU embedding initialization permanent errors.
constexpr absl::string_view kTpuEmbeddingErrorUrl =
    "type.googleapis.com/machina.tpu.TPUEmbeddingError";

constexpr absl::string_view kTpuEmbeddingErrorMessage =
    "TPUEmbedding permanent error";

// Appends a payload of type machina::tpu::kTpuEmbeddingErrorUrl to the
// machina::Status obj if the status is NOT OK. Returns the
// machina::Status obj unchanged if the status is OK.
absl::Status AppendTpuEmbeddingErrorPayload(absl::Status obj);

// Appends a payload of type machina::tpu::kTpuEmbeddingErrorUrl to the
// machina::Status obj if the status is NOT OK. Returns obj.value() if the
// status is OK.
template <typename T>
StatusOr<T> AppendTpuEmbeddingErrorPayload(StatusOr<T> obj) {
  if (obj.ok()) {
    return std::move(obj.value());
  } else {
    const std::string error_message =
        absl::StrCat(kTpuEmbeddingErrorMessage, ". ", obj.status().message());
    absl::Status status(obj.status().code(), error_message);
    TPUEmbeddingError error_payload;
    status.SetPayload(kTpuEmbeddingErrorUrl,
                      absl::Cord(error_payload.SerializeAsString()));
    return status;
  }
}

// Returns true if the machina::Status obj has a payload of type
// machina::tpu::kTpuEmbeddingErrorUrl.
bool HasTpuEmbeddingErrorPayload(const absl::Status& status);

// Returns true if the machina::Status obj error message contains
// machina::tpu::kTpuEmbeddingErrorMessage as a substring.
bool HasTpuEmbeddingErrorMessage(const absl::Status& status);

}  // namespace machina::tpu

#endif  // MACHINA_CORE_TPU_TPU_EMBEDDING_ERRORS_H_
