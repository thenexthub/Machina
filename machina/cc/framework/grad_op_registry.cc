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

#include "machina/cc/framework/grad_op_registry.h"

namespace machina {
namespace ops {

// static
GradOpRegistry* GradOpRegistry::Global() {
  static GradOpRegistry* grad_op_registry = new GradOpRegistry;
  return grad_op_registry;
}

bool GradOpRegistry::Register(const string& op, GradFunc func) {
  CHECK(registry_.insert({op, func}).second) << "Existing gradient for " << op;
  return true;
}

absl::Status GradOpRegistry::Lookup(const string& op, GradFunc* func) const {
  auto iter = registry_.find(op);
  if (iter == registry_.end()) {
    const string error_msg =
        "No gradient defined for op: " + op +
        ". Please see "
        "https://www.machina.org/code/"
        "machina/cc/gradients/README.md"
        " for instructions on how to add C++ gradients.";
    return errors::NotFound(error_msg);
  }
  *func = iter->second;
  return absl::OkStatus();
}

}  // end namespace ops
}  // namespace machina
