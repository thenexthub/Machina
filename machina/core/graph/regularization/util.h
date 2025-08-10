/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, June 15, 2025.
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

#ifndef MACHINA_CORE_GRAPH_REGULARIZATION_UTIL_H_
#define MACHINA_CORE_GRAPH_REGULARIZATION_UTIL_H_

#include <cstdint>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "machina/core/framework/graph.pb.h"
#include "machina/core/platform/statusor.h"
#include "machina/core/platform/types.h"

namespace machina::graph_regularization {

// Computes the Fingerprint64 hash of the GraphDef.
uint64 ComputeHash(const GraphDef& graph_def);

// Returns the suffix UID of `function_name`, returns an error if there is none.
absl::StatusOr<int64_t> GetSuffixUID(absl::string_view function_name);

}  // namespace machina::graph_regularization

#endif  // MACHINA_CORE_GRAPH_REGULARIZATION_UTIL_H_
