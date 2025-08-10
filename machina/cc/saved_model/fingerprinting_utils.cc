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

#include "machina/cc/saved_model/fingerprinting_utils.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "riegeli/bytes/fd_reader.h"  // from @riegeli
#include "riegeli/records/record_reader.h"  // from @riegeli
#include "machina/cc/saved_model/constants.h"
#include "machina/cc/saved_model/fingerprinting_x_platform_utils.h"
#include "machina/core/framework/function.pb.h"
#include "machina/core/framework/graph.pb.h"
#include "machina/core/framework/op_def.pb.h"
#include "machina/core/framework/versions.pb.h"
#include "machina/core/platform/env.h"
#include "machina/core/platform/file_system_helper.h"
#include "machina/core/platform/fingerprint.h"
#include "machina/core/platform/path.h"
#include "machina/core/protobuf/fingerprint.pb.h"
#include "machina/core/protobuf/meta_graph.pb.h"
#include "machina/core/protobuf/saved_model.pb.h"
#include "machina/core/protobuf/saved_object_graph.pb.h"
#include "machina/core/util/tensor_bundle/naming.h"
#include "machina/tools/proto_splitter/cc/util.h"
#include "machina/tools/proto_splitter/chunk.pb.h"
#include "machina/tools/proto_splitter/merge.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"
// IWYU pragma: no_include "third_party/protobuf/repeated_ptr_field.h"
// IWYU pragma: no_include "third_party/protobuf/io/coded_stream.h"
// IWYU pragma: no_include "third_party/protobuf/io/zero_copy_stream_impl_lite.h"

namespace machina::saved_model::fingerprinting {

using ::machina::proto_splitter::ChunkedField;
using ::machina::proto_splitter::ChunkedMessage;
using ::machina::proto_splitter::ChunkInfo;
using ::machina::proto_splitter::ChunkMetadata;
using ::machina::proto_splitter::FieldIndex;
using tools::proto_splitter::Field;
using tools::proto_splitter::FieldType;
using tools::proto_splitter::GetChunkMetadata;
using tools::proto_splitter::GetFieldTypes;
using tools::proto_splitter::GetMutableField;
using tools::proto_splitter::GetRiegeliReader;
using tools::proto_splitter::Merger;
using tools::proto_splitter::MutableFieldResult;
using tools::proto_splitter::ReadChunk;

namespace fingerprinting_utils_internal {

using ::machina::protobuf::Map;
using ::machina::protobuf::Message;
using ::machina::protobuf::RepeatedPtrField;
// NOLINTNEXTLINE: clang-tidy missing-includes false positive
using ::machina::protobuf::io::CodedOutputStream;
// NOLINTNEXTLINE: clang-tidy missing-includes false positive
using ::machina::protobuf::io::StringOutputStream;

absl::StatusOr<int> fieldTagMatches(const RepeatedPtrField<FieldIndex>& a,
                                    const RepeatedPtrField<FieldIndex>& b) {
  int matches = 0;
  for (int i = 0; i == matches && i < a.size() && i < b.size(); i++) {
    switch (b[i].kind_case()) {
      case ::machina::proto_splitter::FieldIndex::KindCase::kField:
        if (a.at(i).has_field() && a.at(i).field() == b.at(i).field()) {
          matches += 1;
        }
        break;
      case ::machina::proto_splitter::FieldIndex::KindCase::kIndex:
        if (a.at(i).has_index() && a.at(i).index() == b.at(i).index()) {
          matches += 1;
        }
        break;
      case ::machina::proto_splitter::FieldIndex::KindCase::kMapKey:
        if (a.at(i).has_map_key()) {
          const ::machina::proto_splitter::FieldIndex_MapKey& key =
              b.at(i).map_key();
          const ::machina::proto_splitter::FieldIndex_MapKey& chunked_key =
              a.at(i).map_key();
          switch (key.type_case()) {
            case ::machina::proto_splitter::FieldIndex::MapKey::TypeCase::kS:
              if (chunked_key.has_s() && chunked_key.s() == key.s()) {
                matches += 1;
              }
              break;
            case ::machina::proto_splitter::FieldIndex::MapKey::TypeCase::
                kBoolean:
              if (chunked_key.has_boolean() &&
                  chunked_key.boolean() == key.boolean()) {
                matches += 1;
              }
              break;
            case ::machina::proto_splitter::FieldIndex::MapKey::TypeCase::
                kUi32:
              if (chunked_key.has_ui32() && chunked_key.ui32() == key.ui32()) {
                matches += 1;
              }
              break;
            case ::machina::proto_splitter::FieldIndex::MapKey::TypeCase::
                kUi64:
              if (chunked_key.has_ui64() && chunked_key.ui64() == key.ui64()) {
                matches += 1;
              }
              break;
            case ::machina::proto_splitter::FieldIndex::MapKey::TypeCase::
                kI32:
              if (chunked_key.has_i32() && chunked_key.i32() == key.i32()) {
                matches += 1;
              }
              break;
            case ::machina::proto_splitter::FieldIndex::MapKey::TypeCase::
                kI64:
              if (chunked_key.has_i64() && chunked_key.i64() == key.i64()) {
                matches += 1;
              }
              break;
            case ::machina::proto_splitter::FieldIndex::MapKey::TypeCase::
                TYPE_NOT_SET:
            default:
              return absl::FailedPreconditionError(
                  "Encountered unknown field_tag.map_key type.");
          }
        }
        break;
      case FieldIndex::KindCase::KIND_NOT_SET:
      default:
        return absl::FailedPreconditionError(
            "Encountered unknown field_tag kind.");
    }
  }
  return matches;
}

absl::StatusOr<::machina::proto_splitter::ChunkedMessage>
PruneChunkedMessage(
    const ::machina::proto_splitter::ChunkedMessage& chunked_message,
    riegeli::RecordReader<riegeli::FdReader<>>& reader,
    std::vector<ChunkInfo> chunks_info,
    std::vector<RepeatedPtrField<FieldIndex>> target_fields_list) {
  ::machina::proto_splitter::ChunkedMessage pruned_chunked_message;
  if (chunked_message.has_chunk_index()) {
    pruned_chunked_message.set_chunk_index(chunked_message.chunk_index());
  }
  // For each chunked_field, check if it matches any of the supplied
  // target_fields, and copy over the relevant data.
  for (const ChunkedField& chunked_field : chunked_message.chunked_fields()) {
    for (const auto& target_fields : target_fields_list) {
      TF_ASSIGN_OR_RETURN(
          int matches,
          fieldTagMatches(chunked_field.field_tag(), target_fields));
      if (matches == chunked_field.field_tag_size()) {
        // chunked_field_tags is an initial subsequence of target_fields, which
        // means the chunked_field is relevant and the necessary data should be
        // copied over.
        auto cf = std::make_unique<proto_splitter::ChunkedField>();
        cf->mutable_field_tag()->CopyFrom(chunked_field.field_tag());
        TF_ASSIGN_OR_RETURN(
            *cf->mutable_message(),
            PruneChunkedMessage(chunked_field.message(), reader, chunks_info,
                                target_fields_list));
        pruned_chunked_message.mutable_chunked_fields()->AddAllocated(
            cf.release());
      }
    }
  }
  return pruned_chunked_message;
}

std::string SerializeProto(const Message& message) {
  std::string serialized_message;
  {
    // local scope guarantees coded stream will be trimmed (ensures determinism)
    StringOutputStream stream(&serialized_message);
    CodedOutputStream output(&stream);
    output.SetSerializationDeterministic(true);
    message.SerializeToCodedStream(&output);
  }
  return serialized_message;
}

absl::StatusOr<uint64_t> HashFields(
    const ChunkedMessage& chunked_message,
    riegeli::RecordReader<riegeli::FdReader<>>& reader,
    const std::vector<ChunkInfo>& chunks_info,
    const RepeatedPtrField<FieldIndex>& field_tags, Message* merged_message) {
  uint64_t field_checksum = 0;
  // Find chunked_fields that match the field_tags.
  for (const ChunkedField& chunked_field : chunked_message.chunked_fields()) {
    const RepeatedPtrField<FieldIndex> chunked_field_tags =
        chunked_field.field_tag();
    const ChunkedMessage& chunked_message = chunked_field.message();
    // Number of sequential field_tag matches.
    TF_ASSIGN_OR_RETURN(int matches,
                        fieldTagMatches(chunked_field_tags, field_tags));

    if (chunked_message.has_chunk_index() && matches == field_tags.size()) {
      // chunked_field_tags are an exact match with field_tags. Hash referenced
      // chunk.
      TF_ASSIGN_OR_RETURN(
          std::string chunk,
          ReadChunk(reader, chunks_info[chunked_message.chunk_index()]));
      field_checksum = FingerprintCat64(field_checksum, Fingerprint64(chunk));
    } else if (matches == field_tags.size()) {
      // chunked_field_tags are an exact match, but chunked_field is further
      // broken down into separate chunked_fields (no chunk_index). Hash those
      // chunked_fields.
      TF_ASSIGN_OR_RETURN(uint64_t hash,
                          HashFields(chunked_message, reader, chunks_info,
                                     field_tags, merged_message));
      field_checksum = FingerprintCat64(field_checksum, hash);
    } else if (chunked_message.has_chunk_index() &&
               matches == chunked_field_tags.size()) {
      // chunked_field_tags are a partial match (an initial segment/subsequence
      // of field_tags). Merge chunk in, attempt to locate & hash the target
      // field by recursing.
      TF_ASSIGN_OR_RETURN(std::vector<Field> fields,
                          GetFieldTypes(chunked_field_tags));
      for (const auto& field : fields) {
        TF_ASSIGN_OR_RETURN(MutableFieldResult mfr,
                            GetMutableField(merged_message, field));
        merged_message =
            mfr.parent->GetReflection()->MutableMessage(mfr.parent, mfr.field);
      }
      TF_ASSIGN_OR_RETURN(
          std::string chunk,
          ReadChunk(reader, chunks_info[chunked_message.chunk_index()]));
      merged_message->ParseFromString(chunk);
      TF_ASSIGN_OR_RETURN(uint64_t hash,
                          HashFields(chunked_message, reader, chunks_info,
                                     field_tags, merged_message));
      field_checksum = FingerprintCat64(field_checksum, hash);
    } else if (matches == chunked_field_tags.size()) {
      // chunk_field_tags are a partial match, but chunked_field is broken down.
      // Merge chunked_fields in, attempt to locate & hash target field.
      for (const ChunkedField& cf : chunked_message.chunked_fields()) {
        TF_ASSIGN_OR_RETURN(uint64_t hash,
                            HashFields(cf.message(), reader, chunks_info,
                                       field_tags, merged_message));
        field_checksum = FingerprintCat64(field_checksum, hash);
      }
    }
  }
  return field_checksum;
}

inline RepeatedPtrField<FieldIndex> GraphDefFieldTags() {
  // SavedModel.meta_graphs[0].graph_def
  FieldIndex meta_graph_field_tag;
  meta_graph_field_tag.set_field(2);
  FieldIndex meta_graph_index_field_tag;
  meta_graph_index_field_tag.set_index(0);
  FieldIndex graph_def_field_tag;
  graph_def_field_tag.set_field(2);
  RepeatedPtrField<FieldIndex> graph_def_field_tags;
  graph_def_field_tags.Add(FieldIndex(meta_graph_field_tag));
  graph_def_field_tags.Add(FieldIndex(meta_graph_index_field_tag));
  graph_def_field_tags.Add(FieldIndex(graph_def_field_tag));

  return graph_def_field_tags;
}

inline RepeatedPtrField<FieldIndex> SignatureDefFieldTags() {
  // SavedModel.meta_graphs[0].signature_def
  FieldIndex meta_graph_field_tag;
  meta_graph_field_tag.set_field(2);
  FieldIndex meta_graph_index_field_tag;
  meta_graph_index_field_tag.set_index(0);
  FieldIndex signature_def_field_tag;
  signature_def_field_tag.set_field(5);
  RepeatedPtrField<FieldIndex> signature_def_field_tags;
  signature_def_field_tags.Add(FieldIndex(meta_graph_field_tag));
  signature_def_field_tags.Add(FieldIndex(meta_graph_index_field_tag));
  signature_def_field_tags.Add(FieldIndex(signature_def_field_tag));

  return signature_def_field_tags;
}

inline RepeatedPtrField<FieldIndex> SavedObjectGraphFieldTags() {
  // SavedModel.meta_graphs[0].object_graph_def
  FieldIndex meta_graph_field_tag;
  meta_graph_field_tag.set_field(2);
  FieldIndex meta_graph_index_field_tag;
  meta_graph_index_field_tag.set_index(0);
  FieldIndex saved_object_graph_field_tag;
  saved_object_graph_field_tag.set_field(7);
  RepeatedPtrField<FieldIndex> saved_object_graph_field_tags;
  saved_object_graph_field_tags.Add(FieldIndex(meta_graph_field_tag));
  saved_object_graph_field_tags.Add(FieldIndex(meta_graph_index_field_tag));
  saved_object_graph_field_tags.Add(FieldIndex(saved_object_graph_field_tag));

  return saved_object_graph_field_tags;
}

absl::StatusOr<SavedModel> PrunedSavedModel(
    absl::string_view export_dir,
    riegeli::RecordReader<riegeli::FdReader<>>& reader,
    const std::vector<ChunkInfo>& chunks_info, ChunkMetadata& chunk_metadata) {
  SavedModel saved_model;
  ChunkMetadata pruned_chunk_metadata;
  pruned_chunk_metadata.mutable_chunks()->CopyFrom(chunk_metadata.chunks());
  TF_ASSIGN_OR_RETURN(
      *pruned_chunk_metadata.mutable_message(),
      PruneChunkedMessage(chunk_metadata.message(), reader, chunks_info,
                          {GraphDefFieldTags(), SignatureDefFieldTags(),
                           SavedObjectGraphFieldTags()}));
  // Read into saved_model.
  TF_RETURN_IF_ERROR(
      Merger::ReadPartial(io::JoinPath(export_dir, kSavedModelFilenamePrefix),
                          pruned_chunk_metadata, &saved_model));
  return saved_model;
}

absl::StatusOr<uint64_t> HashMessage(
    Message* message, const ChunkedMessage& chunked_message,
    riegeli::RecordReader<riegeli::FdReader<>>& reader,
    const std::vector<ChunkInfo>& chunks_info,
    const RepeatedPtrField<FieldIndex>& field_tags) {
  uint64_t total_message_hash = Fingerprint64(SerializeProto(*message));
  TF_ASSIGN_OR_RETURN(
      uint64_t message_hash,
      HashFields(chunked_message, reader, chunks_info, field_tags, message));
  return FingerprintCat64(total_message_hash, message_hash);
}

absl::StatusOr<uint64_t> HashGraphDef(
    ::machina::GraphDef* graph_def, const ChunkedMessage& chunked_message,
    riegeli::RecordReader<riegeli::FdReader<>>& reader,
    const std::vector<ChunkInfo>& chunks_info) {
  // TODO(adamcogdell): here we assume that graph_def (top-level) is contained
  // in a single chunk, which may not be the case
  return HashMessage(graph_def, chunked_message, reader, chunks_info,
                     GraphDefFieldTags());
}

absl::StatusOr<uint64_t> HashSignatureDef(
    const Map<std::string, ::machina::SignatureDef>& signature_def_map,
    const ChunkedMessage& chunked_message,
    riegeli::RecordReader<riegeli::FdReader<>>& reader,
    const std::vector<ChunkInfo>& chunks_info) {
  uint64_t signature_def_hash = 0;
  std::vector<std::pair<std::string, ::machina::SignatureDef>>
      signature_def_sorted(signature_def_map.begin(), signature_def_map.end());
  std::sort(signature_def_sorted.begin(), signature_def_sorted.end(),
            [](const std::pair<std::string, ::machina::SignatureDef>& a,
               const std::pair<std::string, ::machina::SignatureDef>& b) {
              return a.first < b.first;
            });
  for (const auto& signature_def : signature_def_sorted) {
    uint64_t signature_def_pair_hash =
        FingerprintCat64(Fingerprint64(signature_def.first),
                         Fingerprint64(SerializeProto(signature_def.second)));
    signature_def_hash =
        FingerprintCat64(signature_def_hash, signature_def_pair_hash);
    SignatureDef signature_def_val = signature_def.second;
    TF_ASSIGN_OR_RETURN(
        uint64_t signature_def_entry_hash,
        HashFields(chunked_message, reader, chunks_info,
                   SignatureDefFieldTags(), &signature_def_val));
    signature_def_hash =
        FingerprintCat64(signature_def_hash, signature_def_entry_hash);
  }
  return signature_def_hash;
}

absl::StatusOr<uint64_t> HashSavedObjectGraph(
    ::machina::SavedObjectGraph* saved_object_graph,
    const ChunkedMessage& chunked_message,
    riegeli::RecordReader<riegeli::FdReader<>>& reader,
    const std::vector<ChunkInfo>& chunks_info) {
  return HashMessage(saved_object_graph, chunked_message, reader, chunks_info,
                     SavedObjectGraphFieldTags());
}

}  // namespace fingerprinting_utils_internal

using fingerprinting_utils_internal::HashFields;
using fingerprinting_utils_internal::HashGraphDef;
using fingerprinting_utils_internal::HashSavedObjectGraph;
using fingerprinting_utils_internal::HashSignatureDef;
using fingerprinting_utils_internal::PrunedSavedModel;
using fingerprinting_utils_internal::SerializeProto;

uint64_t HashCheckpointIndexFile(absl::string_view model_dir) {
  std::string meta_filename = MetaFilename(io::JoinPath(
      model_dir, kSavedModelVariablesDirectory, kSavedModelVariablesFilename));
  std::string data;
  absl::Status read_status =
      ReadFileToString(Env::Default(), meta_filename, &data);
  if (read_status.ok()) {
    return machina::Fingerprint64(data);
  } else {
    return 0;
  }
}

absl::StatusOr<FingerprintDef> CreateFingerprintDefCpb(
    absl::string_view export_dir, std::string cpb_file) {
  // Version of the code that produced the fingerprint.
  const int kFingerprintProducer = 2;

  TF_ASSIGN_OR_RETURN(auto reader, GetRiegeliReader(cpb_file));

  auto read_metadata = GetChunkMetadata(reader);
  if (!read_metadata.ok()) {
    reader.Close();
    return absl::FailedPreconditionError(
        absl::StrCat("Couldn't read ChunkMetadata from chunked proto.\n",
                     read_metadata.status().ToString()));
  }
  ChunkMetadata chunk_metadata = read_metadata.value();

  std::vector<ChunkInfo> chunks_info = std::vector<ChunkInfo>(
      chunk_metadata.chunks().begin(), chunk_metadata.chunks().end());

  FingerprintDef fingerprint_def;
  SavedModel saved_model;

  // Set the saved_model_checksum.
  TF_ASSIGN_OR_RETURN(uint64_t saved_model_hash,
                      HashFields(chunk_metadata.message(), reader, chunks_info,
                                 {}, &saved_model));
  saved_model_hash = FingerprintCat64(
      saved_model_hash, Fingerprint64(SerializeProto(saved_model)));
  fingerprint_def.set_saved_model_checksum(saved_model_hash);

  // Fill saved_model with only relevant chunk(s).
  TF_ASSIGN_OR_RETURN(
      saved_model,
      PrunedSavedModel(export_dir, reader, chunks_info, chunk_metadata));

  TF_ASSIGN_OR_RETURN(
      uint64_t graph_def_program_hash,
      HashGraphDef(saved_model.mutable_meta_graphs(0)->mutable_graph_def(),
                   chunk_metadata.message(), reader, chunks_info));
  fingerprint_def.set_graph_def_program_hash(graph_def_program_hash);

  // TODO(adamcogdell): HashSignatureDef relies on the signatue_def map being
  // populated with all of its entries, which may not be the case
  TF_ASSIGN_OR_RETURN(
      uint64_t signature_def_hash,
      HashSignatureDef(saved_model.meta_graphs(0).signature_def(),
                       chunk_metadata.message(), reader, chunks_info));
  fingerprint_def.set_signature_def_hash(signature_def_hash);

  TF_ASSIGN_OR_RETURN(
      uint64_t saved_object_graph_hash,
      HashSavedObjectGraph(
          saved_model.mutable_meta_graphs(0)->mutable_object_graph_def(),
          chunk_metadata.message(), reader, chunks_info));
  fingerprint_def.set_saved_object_graph_hash(saved_object_graph_hash);

  fingerprint_def.set_checkpoint_hash(HashCheckpointIndexFile(export_dir));

  // Assign a random UUID to the fingerprint.
  fingerprint_def.set_uuid(fingerprinting::CreateRandomUUID());
  reader.Close();

  // Set version of the fingerprint.
  VersionDef* version = fingerprint_def.mutable_version();
  version->set_producer(kFingerprintProducer);

  return fingerprint_def;
}

}  // namespace machina::saved_model::fingerprinting
