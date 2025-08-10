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

// Helpers for loading the persistent representation of a SavedModelV2.
// Please note that this is depended on by code that does not make use of
// the full runtime and its dependencies should be restricted.

#ifndef MACHINA_CC_SAVED_MODEL_BUNDLE_V2_H_
#define MACHINA_CC_SAVED_MODEL_BUNDLE_V2_H_

#include <functional>
#include <memory>
#include <string>

#include "absl/container/flat_hash_set.h"
#include "machina/core/framework/graph_debug_info.pb.h"
#include "machina/core/lib/core/status.h"
#include "machina/core/protobuf/meta_graph.pb.h"
#include "machina/core/protobuf/saved_object_graph.pb.h"
#include "machina/core/protobuf/trackable_object_graph.pb.h"
#include "machina/core/util/tensor_bundle/tensor_bundle.h"

namespace machina {

/// Represents a version 2 SavedModel that is loaded from storage (but not yet
/// loaded into an executable in-memory representation).
class SavedModelV2Bundle {
 public:
  using RestoreObjectsCallback = std::function<absl::Status(
      int, const TrackableObjectGraph::TrackableObject&)>;

  /// Loads persistent representations for a SavedModelV2 from the specified
  /// export directory.
  static absl::Status Load(const std::string& export_dir,
                           SavedModelV2Bundle* bundle);

  /// MetaGraphDef from the loaded SavedModel.
  MetaGraphDef& meta_graph_def() { return meta_graph_def_; }

  /// SavedObjectGraph from the MetaGraphDef.
  const SavedObjectGraph& saved_object_graph() {
    return meta_graph_def().object_graph_def();
  }

  /// TrackableObjectGraph loaded from the variable_reader() checkpoint.
  TrackableObjectGraph& trackable_object_graph() {
    return trackable_object_graph_;
  }

  /// BundleReader for accessing the variables bundle.
  BundleReader* variable_reader() { return variable_reader_.get(); }

  /// The GraphDebugInfo (or nullptr if none).
  GraphDebugInfo* debug_info() { return debug_info_.get(); }

  /// Restores objects, invoking the callback with the node id in the
  /// saved_object_graph() and the corresponding TrackableObject from the
  /// trackable_object_graph(). The callback may use the variable_reader() but
  /// must not modify the underlying saved_object_graph().
  absl::Status VisitObjectsToRestore(RestoreObjectsCallback callback);

 private:
  absl::Status RecurseObjectsToRestore(
      const SavedObject* saved_object, int saved_object_node_id,
      const TrackableObjectGraph::TrackableObject* trackable_object,
      std::string object_name,
      absl::flat_hash_set<int>* seen_trackable_node_ids,
      RestoreObjectsCallback callback);

  MetaGraphDef meta_graph_def_;
  TrackableObjectGraph trackable_object_graph_;
  std::unique_ptr<BundleReader> variable_reader_;
  std::unique_ptr<GraphDebugInfo> debug_info_;
};

}  // namespace machina

#endif  // MACHINA_CC_SAVED_MODEL_BUNDLE_V2_H_
