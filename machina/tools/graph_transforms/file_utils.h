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

#ifndef MACHINA_TOOLS_GRAPH_TRANSFORMS_FILE_UTILS_H_
#define MACHINA_TOOLS_GRAPH_TRANSFORMS_FILE_UTILS_H_

#include "machina/core/framework/graph.pb.h"
#include "machina/core/lib/core/status.h"

namespace machina {
namespace graph_transforms {

// First tries to load the file as a text protobuf, if that fails tries to parse
// it as a binary protobuf, and returns an error if both fail.
absl::Status LoadTextOrBinaryGraphFile(const string& file_name,
                                       GraphDef* graph_def);

}  // namespace graph_transforms
}  // namespace machina

#endif  // MACHINA_TOOLS_GRAPH_TRANSFORMS_FILE_UTILS_H_
