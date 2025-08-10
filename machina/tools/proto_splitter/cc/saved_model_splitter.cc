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
#include "machina/tools/proto_splitter/cc/saved_model_splitter.h"

#include <vector>

#include "absl/status/status.h"
#include "machina/core/framework/graph.pb.h"
#include "machina/core/protobuf/meta_graph.pb.h"
#include "machina/core/protobuf/saved_model.pb.h"
#include "machina/tools/proto_splitter/cc/graph_def_splitter.h"
#include "machina/tools/proto_splitter/cc/large_node_splitter.h"
#include "machina/tools/proto_splitter/cc/max_size.h"
#include "machina/tools/proto_splitter/cc/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/protobuf.h"

namespace machina {
namespace tools::proto_splitter {

// Required in OSS to prevent string to bool conversion in FieldType variant.
using namespace std::string_literals;  // NOLINT

absl::Status SavedModelSplitter::BuildChunks() {
  TF_RETURN_IF_ERROR(SetMessageAsBaseChunk());
  SavedModel* sm = tsl::protobuf::DynamicCastToGenerated<SavedModel>(message());
  int max_size = GetMaxSize();
  if (GetInitialSize() < max_size) return absl::OkStatus();

  std::vector<FieldType> fields_to_graph_def = {"meta_graphs"s, 0,
                                                "graph_def"s};
  GraphDefSplitter graph_def_splitter(
      sm->mutable_meta_graphs(0)->mutable_graph_def(), this,
      &fields_to_graph_def);
  TF_RETURN_IF_ERROR(graph_def_splitter.BuildChunks());

  if (sm->ByteSizeLong() < max_size) return absl::OkStatus();

  LargeNodeSplitter<GraphDef> entire_graph_splitter(
      sm->mutable_meta_graphs(0)->mutable_graph_def(), this,
      &fields_to_graph_def);
  int index = 1;
  entire_graph_splitter.SetChunkIndex(&index);
  TF_RETURN_IF_ERROR(entire_graph_splitter.BuildChunks());

  return absl::OkStatus();
}

}  // namespace tools::proto_splitter
}  // namespace machina
