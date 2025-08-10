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

#ifndef MACHINA_CC_SAVED_MODEL_FINGERPRINTING_H_
#define MACHINA_CC_SAVED_MODEL_FINGERPRINTING_H_

#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "machina/core/protobuf/fingerprint.pb.h"

namespace machina::saved_model::fingerprinting {

// Creates a FingerprintDef proto from a SavedModel (regular or chunked) and the
// checkpoint meta file (.index) in `export_dir`.
absl::StatusOr<FingerprintDef> CreateFingerprintDef(
    absl::string_view export_dir);

// Loads the `fingerprint.pb` from `export_dir`, returns an error if there is
// none.
absl::StatusOr<FingerprintDef> ReadSavedModelFingerprint(
    absl::string_view export_dir);

// Canonical fingerprinting ID for a SavedModel.
std::string Singleprint(uint64_t graph_def_program_hash,
                        uint64_t signature_def_hash,
                        uint64_t saved_object_graph_hash,
                        uint64_t checkpoint_hash);
std::string Singleprint(const FingerprintDef& fingerprint);
absl::StatusOr<std::string> Singleprint(absl::string_view export_dir);

}  // namespace machina::saved_model::fingerprinting

#endif  // MACHINA_CC_SAVED_MODEL_FINGERPRINTING_H_
