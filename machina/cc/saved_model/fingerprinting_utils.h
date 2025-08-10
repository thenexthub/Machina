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

#ifndef MACHINA_CC_SAVED_MODEL_FINGERPRINTING_UTILS_H_
#define MACHINA_CC_SAVED_MODEL_FINGERPRINTING_UTILS_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "riegeli/bytes/fd_reader.h"  // from @riegeli
#include "riegeli/records/record_reader.h"  // from @riegeli
#include "machina/core/framework/graph.pb.h"
#include "machina/core/platform/protobuf.h"  // IWYU pragma: keep
#include "machina/core/protobuf/fingerprint.pb.h"
#include "machina/core/protobuf/meta_graph.pb.h"
#include "machina/core/protobuf/saved_model.pb.h"
#include "machina/core/protobuf/saved_object_graph.pb.h"
#include "machina/tools/proto_splitter/chunk.pb.h"

namespace machina::saved_model::fingerprinting {

namespace fingerprinting_utils_internal {

using ::machina::protobuf::Map;
using ::machina::protobuf::Message;
using ::machina::protobuf::RepeatedPtrField;

// Number of sequential FieldIndex matches of `a` in `b`. (Length of initial
// subsequence.)
// Example: `a = {4, 2}`, `b = {4, 2, 1, 3}`, `fieldTagMatches(a, b) == 2`
absl::StatusOr<int> fieldTagMatches(
    const RepeatedPtrField<::machina::proto_splitter::FieldIndex>& a,
    const RepeatedPtrField<::machina::proto_splitter::FieldIndex>& b);

// Pull out the relevant data within `chunked_message`. A `chunked_field` is
// relevant if its `field_tags` are an initial subsequence any of the
// `target_fields` in the provided `target_fields_list`.
absl::StatusOr<::machina::proto_splitter::ChunkedMessage>
PruneChunkedMessage(
    const ::machina::proto_splitter::ChunkedMessage& chunked_message,
    riegeli::RecordReader<riegeli::FdReader<>>& reader,
    std::vector<::machina::proto_splitter::ChunkInfo> chunks_info,
    std::vector<RepeatedPtrField<::machina::proto_splitter::FieldIndex>>
        target_fields_list);

// Deterministically serializes the proto `message`.
std::string SerializeProto(const Message& message);

// Uses metadata contained in `chunked_message` to hash fields within the
// data accessed by the `reader` using `chunks_info`.
absl::StatusOr<uint64_t> HashFields(
    const ::machina::proto_splitter::ChunkedMessage& chunked_message,
    riegeli::RecordReader<riegeli::FdReader<>>& reader,
    const std::vector<::machina::proto_splitter::ChunkInfo>& chunks_info,
    const RepeatedPtrField<::machina::proto_splitter::FieldIndex>&
        field_tags,
    Message* merged_message);

// Gets the field tags for `graph_def`.::machina
inline RepeatedPtrField<::machina::proto_splitter::FieldIndex>
GraphDefFieldTags();

// Gets the field tags for `signature_def`.
inline RepeatedPtrField<::machina::proto_splitter::FieldIndex>
SignatureDefFieldTags();

// Gets the field tags for `saved_object_graph`.
inline RepeatedPtrField<::machina::proto_splitter::FieldIndex>
SavedObjectGraphFieldTags();

// Returns a `SavedModel` containing only fields (up to those) specified by
// `GraphDefFieldTags()`, `SignatureDefFieldTags()`, and
// `SavedObjectGraphFieldTags()`.
absl::StatusOr<machina::SavedModel> PrunedSavedModel(
    absl::string_view export_dir,
    riegeli::RecordReader<riegeli::FdReader<>>& reader,
    const std::vector<::machina::proto_splitter::ChunkInfo>& chunks_info,
    ::machina::proto_splitter::ChunkMetadata& chunk_metadata);

// Hashes the contents of `message` specified by `field_tags`.
absl::StatusOr<uint64_t> HashMessage(
    Message* message,
    const ::machina::proto_splitter::ChunkedMessage& chunked_message,
    riegeli::RecordReader<riegeli::FdReader<>>& reader,
    const std::vector<::machina::proto_splitter::ChunkInfo>& chunks_info,
    const RepeatedPtrField<::machina::proto_splitter::FieldIndex>&
        field_tags);

// Hashes the contents of `graph_def`.
absl::StatusOr<uint64_t> HashGraphDef(
    machina::GraphDef* graph_def,
    const ::machina::proto_splitter::ChunkedMessage& chunked_message,
    riegeli::RecordReader<riegeli::FdReader<>>& reader,
    const std::vector<::machina::proto_splitter::ChunkInfo>& chunks_info);

// Hashes the contents of `signature_def`.
absl::StatusOr<uint64_t> HashSignatureDef(
    const Map<std::string, ::machina::SignatureDef>& signature_def_map,
    const ::machina::proto_splitter::ChunkedMessage& chunked_message,
    riegeli::RecordReader<riegeli::FdReader<>>& reader,
    const std::vector<::machina::proto_splitter::ChunkInfo>& chunks_info);

// Hashes the contents of `saved_object_graph`.
absl::StatusOr<uint64_t> HashSavedObjectGraph(
    machina::SavedObjectGraph* saved_object_graph,
    const ::machina::proto_splitter::ChunkedMessage& chunked_message,
    riegeli::RecordReader<riegeli::FdReader<>>& reader,
    const std::vector<::machina::proto_splitter::ChunkInfo>& chunks_info);

}  // namespace fingerprinting_utils_internal

// Returns the hash of the checkpoint .index file, 0 if there is none.
uint64_t HashCheckpointIndexFile(absl::string_view model_dir);

// Creates a FingerprintDef proto from a chunked SavedModel and the checkpoint
// meta file (.index) in `export_dir`.
absl::StatusOr<FingerprintDef> CreateFingerprintDefCpb(
    absl::string_view export_dir, std::string cpb_file);

}  // namespace machina::saved_model::fingerprinting

#endif  // MACHINA_CC_SAVED_MODEL_FINGERPRINTING_UTILS_H_
