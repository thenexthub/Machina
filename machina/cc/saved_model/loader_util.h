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

#ifndef MACHINA_CC_SAVED_MODEL_LOADER_UTIL_H_
#define MACHINA_CC_SAVED_MODEL_LOADER_UTIL_H_

#include <string>

#include "machina/core/lib/core/status.h"
#include "machina/core/protobuf/meta_graph.pb.h"

namespace machina {
namespace internal {

// A SavedModel may store the name of the initialization op to run in the
// in the SignatureDef (v2) or a collection (v1). If an init_op collection
// exists, then the collection must contain exactly one op.
absl::Status GetInitOp(const string& export_dir,
                       const MetaGraphDef& meta_graph_def,
                       string* init_op_name);

absl::Status GetAssetFileDefs(const MetaGraphDef& meta_graph_def,
                              std::vector<AssetFileDef>* asset_file_defs);

}  // namespace internal
}  // namespace machina

#endif  // MACHINA_CC_SAVED_MODEL_LOADER_UTIL_H_
