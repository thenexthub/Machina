/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 29, 2025.
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

/// Functions to read the SavedModel proto, or parts of it.

#ifndef MACHINA_CC_SAVED_MODEL_READER_H_
#define MACHINA_CC_SAVED_MODEL_READER_H_

#include <memory>
#include <unordered_set>

#include "absl/status/statusor.h"
#include "machina/core/framework/graph_debug_info.pb.h"
#include "machina/core/lib/core/status.h"
#include "machina/core/platform/types.h"
#include "machina/core/protobuf/meta_graph.pb.h"
#include "machina/core/protobuf/saved_model.pb.h"

namespace machina {
absl::Status ReadSavedModel(absl::string_view export_dir,
                            SavedModel* saved_model_proto);

// Finds and returns the MetaGraphDef (within the provided SavedModel) that
// matches the given set of tags. The lifetime of the returned MetaGraphDef is
// the same as the lifetime of `saved_model_proto`.
//
// FindMetaGraphDef returns a failure status when no MetaGraphDef matches the
// provided tags.
absl::StatusOr<MetaGraphDef*> FindMetaGraphDef(
    const std::unordered_set<string>& tags, SavedModel* saved_model_proto);

// Reads the SavedModel proto from saved_model.pb(txt) in the given directory,
// finds the MetaGraphDef that matches the given set of tags and writes it to
// the `meta_graph_def` parameter. Returns a failure status when the SavedModel
// file does not exist or no MetaGraphDef matches the tags.
absl::Status ReadMetaGraphDefFromSavedModel(
    absl::string_view export_dir, const std::unordered_set<string>& tags,
    MetaGraphDef* meta_graph_def);

// Store debug info from the SavedModel export dir.
absl::Status ReadSavedModelDebugInfoIfPresent(
    absl::string_view export_dir,
    std::unique_ptr<GraphDebugInfo>* debug_info_proto);

}  // namespace machina

#endif  // MACHINA_CC_SAVED_MODEL_READER_H_
