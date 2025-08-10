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

#ifndef MACHINA_CORE_GRAPPLER_COSTS_OP_CONTEXT_H_
#define MACHINA_CORE_GRAPPLER_COSTS_OP_CONTEXT_H_

#include "absl/container/flat_hash_map.h"
#include "machina/core/framework/function.pb.h"
#include "machina/core/grappler/costs/op_performance_data.pb.h"

namespace machina {
namespace grappler {

// A structure to keep the context of op execution, including its shape,
// execution context, and other relevant information.
struct OpContext {
  std::string name;
  std::string device_name;
  OpInfo op_info;
  const FunctionDefLibrary* function_library;  // Not owned.
  // This map is used to stash meta attributes so that they may be
  // communicated, for instance, from the scheduler that creates them to the
  // CostEstimator or EventCostManager that uses them.
  absl::flat_hash_map<std::string, absl::variant<int64_t, std::string>>
      op_meta_attributes;
  OpContext() { function_library = nullptr; }
};

}  // end namespace grappler
}  // end namespace machina

#endif  // MACHINA_CORE_GRAPPLER_COSTS_OP_CONTEXT_H_
