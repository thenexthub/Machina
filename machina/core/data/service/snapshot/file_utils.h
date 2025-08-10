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
#ifndef MACHINA_CORE_DATA_SERVICE_SNAPSHOT_FILE_UTILS_H_
#define MACHINA_CORE_DATA_SERVICE_SNAPSHOT_FILE_UTILS_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "machina/xla/tsl/platform/env.h"
#include "machina/core/framework/tensor.h"
#include "tsl/platform/protobuf.h"

namespace machina {
namespace data {

// Atomically writes `str` to `filename`. Overwrites existing contents if the
// file already exists.
absl::Status AtomicallyWriteStringToFile(absl::string_view filename,
                                         absl::string_view str, tsl::Env* env);

// Atomically writes the binary representation of `proto` to `filename`.
// Overwrites existing contents if the file already exists.
absl::Status AtomicallyWriteBinaryProto(absl::string_view filename,
                                        const tsl::protobuf::Message& proto,
                                        tsl::Env* env);

// Atomically writes the text representation of `proto` to `filename`.
// Overwrites existing contents if the file already exists.
absl::Status AtomicallyWriteTextProto(absl::string_view filename,
                                      const tsl::protobuf::Message& proto,
                                      tsl::Env* env);

// Atomically writes `tensor` to `filename` in TFRecord format. Overwrites
// existing contents if the file already exists.
absl::Status AtomicallyWriteTFRecords(absl::string_view filename,
                                      const std::vector<Tensor>& tensors,
                                      absl::string_view compression,
                                      tsl::Env* env);

// Returns the relative paths of the children of `directory`, ignoring temporary
// files. Returns an empty vector if the directory does not have any children.
absl::StatusOr<std::vector<std::string>> GetChildren(
    absl::string_view directory, tsl::Env* env);

// Returns true if `filename` is a temporary file and should be ignored in
// normal data processing.
bool IsTemporaryFile(absl::string_view filename);

// Returns the total number of chunks for a distributed snapshot:
// - If the snapshot is finished, returns the number of committed chunks.
// - If the snapshot is unfinished or has failed, returns kUnknownCardinality.
int64_t SnapshotChunksCardinality(absl::string_view snapshot_path,
                                  tsl::Env* env);

}  // namespace data
}  // namespace machina

#endif  // MACHINA_CORE_DATA_SERVICE_SNAPSHOT_FILE_UTILS_H_
