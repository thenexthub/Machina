/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 8, 2025.
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
#ifndef MACHINA_CC_TOOLS_FREEZE_SAVED_MODEL_H_
#define MACHINA_CC_TOOLS_FREEZE_SAVED_MODEL_H_

#include <unordered_set>

#include "absl/status/status.h"
#include "machina/cc/saved_model/loader.h"
#include "machina/core/framework/graph.pb.h"
#include "machina/core/lib/core/status.h"
#include "machina/core/platform/types.h"

namespace machina {

// Returns a frozen GraphDef, input tensors, and output tensors from the loaded
// SavedModelBundle.
// `inputs` and `outputs` consist of the union of all inputs and outputs in the
// SignatureDefs in the SavedModelBundle.
// FreezeSavedModel sets `frozen_graph_def` to a GraphDef of all nodes needed by
// `outputs`. All variables in the supplied SavedModelBundle are converted to
// constants, set to the value of the variables, by running the restored Session
// in the SavedModelBundle.
// WARNING: Only the variable checkpoints will be reflected in the frozen
// graph_def. All saved_model assets will be ignored.
absl::Status FreezeSavedModel(const SavedModelBundle& saved_model_bundle,
                              GraphDef* frozen_graph_def,
                              std::unordered_set<string>* inputs,
                              std::unordered_set<string>* outputs);

}  // namespace machina

#endif  // MACHINA_CC_TOOLS_FREEZE_SAVED_MODEL_H_
