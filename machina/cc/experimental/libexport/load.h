/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Saturday, June 21, 2025.
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
#ifndef MACHINA_CC_EXPERIMENTAL_LIBEXPORT_LOAD_H_
#define MACHINA_CC_EXPERIMENTAL_LIBEXPORT_LOAD_H_

#include <string>

#include "absl/container/flat_hash_map.h"
#include "machina/core/framework/function.pb.h"
#include "machina/core/framework/types.pb.h"
#include "machina/core/platform/protobuf.h"
#include "machina/core/platform/statusor.h"
#include "machina/core/protobuf/saved_model.pb.h"
#include "machina/core/protobuf/saved_object_graph.pb.h"
#include "machina/core/protobuf/trackable_object_graph.pb.h"
#include "machina/core/util/tensor_bundle/tensor_bundle.h"

namespace machina {
namespace libexport {

// A low-level representation of a SavedModel.
//
// This class should only ever be a thin wrapper around disk (or other storage)
// access for a SavedModel.  Higher level functionality should be layered on top
// by other functions and classes.
//
// In the future, this class can also provide a mechanism for automatic version
// migration.  This will allow the calling code to always work against the most
// recent version of SavedModel.
class TFPackage {
 public:
  // Load a SavedModel, parsing the associated protobuf for later access.
  static absl::StatusOr<TFPackage> Load(const std::string& path);

  // Reads and returns a checkpoint key associated with a variable.
  //
  // The variable is identified by the index in the object graph node list.
  //
  // RestoreV2 is the operation that will ultimately be responsible for reading
  // and restoring the variable(s)' values.  Variable values are indexed in the
  // checkpoint files by "checkpoint keys".  These keys along with dtype and
  // shape / slice information allow RestoreV2 to look up a variable's value in
  // the SavedModel and restore it into a tensor.
  absl::StatusOr<std::string> GetVariableCheckpointKey(int index);

  // Retrieves the object graph from the SavedModel.
  //
  // For now, we're returning the object graph directly (i.e. the parsed proto)
  // rather than adding abstraction on top.  We may later find we would like an
  // intermediate abstraction layer to make traversal easier, but for now the
  // extra complexity doesn't seem justified.  Regardless of what we choose,
  // that logic should live outside this class; this class should continue to
  // have the clearly-defined, singular responsibility of reading and parsing
  // the low-level, serialized format.
  const SavedObjectGraph& GetObjectGraph();

  // Retrieves a specific GraphDef node by name.
  //
  // GraphDef nodes are stored as a repeating list of nodes.  At module load
  // time, a module may have constants that need to be restored.  To restore
  // these constants, they are looked up in the GraphDef's nodes by their name.
  // Since we may need to load many constants, we create a hash map of these
  // names to their corresponding nodes at load time in order to look them up
  // in constant time.
  absl::StatusOr<const machina::NodeDef*> GetGraphDefNode(std::string name);

  // Returns a list of function defs in the SavedModel.
  const protobuf::RepeatedPtrField<FunctionDef>& GetFunctionDefs();

  // Returns a BundleReader for reading variable values.
  //
  // This TFPackage retains ownership of the underlying reader.
  machina::BundleReader* GetVariableReader() {
    return variable_reader_.get();
  }

  // Returns whether or not we found a valid checkpoint when loading the
  // package.
  bool HasCheckpoint() { return has_checkpoint_; }

  // Returns the path to the variables file.
  const std::string GetVariablesFilepath() const { return variables_filepath_; }

 private:
  SavedModel saved_model_proto_;
  TrackableObjectGraph trackable_object_graph_;
  std::unique_ptr<machina::BundleReader> variable_reader_;
  std::string variables_filepath_;
  bool has_checkpoint_;
  absl::flat_hash_map<std::string, const NodeDef*> graph_def_nodes_by_name_;
};

}  // namespace libexport
}  // namespace machina

#endif  // MACHINA_CC_EXPERIMENTAL_LIBEXPORT_LOAD_H_
