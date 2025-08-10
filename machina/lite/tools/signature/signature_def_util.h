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
#ifndef MACHINA_LITE_TOOLS_SIGNATURE_SIGNATURE_DEF_UTIL_H_
#define MACHINA_LITE_TOOLS_SIGNATURE_SIGNATURE_DEF_UTIL_H_

#include <map>
#include <string>

#include "absl/status/status.h"
#include "machina/core/lib/core/status.h"
#include "machina/core/platform/status.h"
#include "machina/core/protobuf/meta_graph.pb.h"
#include "machina/lite/core/c/common.h"
#include "machina/lite/schema/schema_generated.h"

namespace tflite {

// Constant for name of the Metadata entry associated with SignatureDefs.
constexpr char kSignatureDefsMetadataName[] = "signature_defs_metadata";

// The function `SetSignatureDefMap()` results in
// `model_data_with_signature_defs` containing a serialized TFLite model
// identical to `model` with a metadata and associated buffer containing
// a FlexBuffer::Map with `signature_def_map` keys and values serialized to
// String.
//
// If a Metadata entry containing a SignatureDef map exists, it will be
//   overwritten.
//
// Returns error if `model_data_with_signature_defs` is null or
//   `signature_def_map` is empty.
//
// On success, returns machina::OkStatus() or error otherwise.
// On error, `model_data_with_signature_defs` is unchanged.
absl::Status SetSignatureDefMap(
    const Model* model,
    const std::map<std::string, machina::SignatureDef>& signature_def_map,
    std::string* model_data_with_signature_defs);

// The function `HasSignatureDef()` returns true if `model` contains a Metadata
// table pointing to a buffer containing a FlexBuffer::Map and the map has
// `signature_key` as a key, or false otherwise.
bool HasSignatureDef(const Model* model, const std::string& signature_key);

// The function `GetSignatureDefMap()` results in `signature_def_map`
// pointing to a map<std::string, machina::SignatureDef>
// parsed from `model`'s metadata buffer.
//
// If the Metadata entry does not exist, `signature_def_map` is unchanged.
// If the Metadata entry exists but cannot be parsed, returns an error.
absl::Status GetSignatureDefMap(
    const Model* model,
    std::map<std::string, machina::SignatureDef>* signature_def_map);

// The function `ClearSignatureDefs` results in `model_data`
// containing a serialized Model identical to `model` omitting any
// SignatureDef-related metadata or buffers.
absl::Status ClearSignatureDefMap(const Model* model, std::string* model_data);

}  // namespace tflite

#endif  // MACHINA_LITE_TOOLS_SIGNATURE_SIGNATURE_DEF_UTIL_H_
