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
#ifndef MACHINA_CC_SAVED_MODEL_UTIL_H_
#define MACHINA_CC_SAVED_MODEL_UTIL_H_

#include <set>
#include <string>
#include <utility>
#include <vector>

#include "machina/core/framework/tensor.h"
#include "machina/core/framework/tensor.pb.h"
#include "machina/core/platform/status.h"
#include "machina/core/protobuf/meta_graph.pb.h"
#include "machina/core/protobuf/saved_model.pb.h"

namespace machina {
namespace saved_model {

// Utility functions for SavedModel reading and writing.

// Returns "WriteVersion" ("1" or "2") of the SavedModel protobuf. If the
// protobuf has exactly one MetaGraphDef, which contains a SavedObjectGraph, it
// is version 2. Else, the protobuf is version 1.
//
// NOTE: The "WriteVersion" does *not* equal the major version of TF.
std::string GetWriteVersion(const SavedModel& saved_model);

// Get view of string keys of a map.
std::set<std::string> GetMapKeys(
    const ::google::protobuf::Map<std::string, ::machina::TensorProto>& map);

// Get the default input value from signature if it's missing in the request
// inputs. If `is_alias` is set to true, the keys of the `request_inputs` are
// alias names rather than the feed names in the graph.
absl::Status GetInputValues(
    const SignatureDef& signature,
    const ::google::protobuf::Map<std::string, ::machina::TensorProto>& request_inputs,
    std::vector<std::pair<string, Tensor>>& inputs);

}  // namespace saved_model
}  // namespace machina

#endif  // MACHINA_CC_SAVED_MODEL_UTIL_H_
