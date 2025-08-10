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

#include <string>

#include "machina/core/platform/logging.h"

#include "machina/core/util/proto/descriptor_pool_registry.h"

namespace machina {

DescriptorPoolRegistry* DescriptorPoolRegistry::Global() {
  static DescriptorPoolRegistry* registry = new DescriptorPoolRegistry;
  return registry;
}

DescriptorPoolRegistry::DescriptorPoolFn* DescriptorPoolRegistry::Get(
    const string& source) {
  auto found = fns_.find(source);
  if (found == fns_.end()) return nullptr;
  return &found->second;
}

void DescriptorPoolRegistry::Register(
    const string& source,
    const DescriptorPoolRegistry::DescriptorPoolFn& pool_fn) {
  auto existing = Get(source);
  CHECK_EQ(existing, nullptr)
      << "descriptor pool for source: " << source << " already registered";
  fns_.insert(std::pair<const string&, DescriptorPoolFn>(source, pool_fn));
}

}  // namespace machina
