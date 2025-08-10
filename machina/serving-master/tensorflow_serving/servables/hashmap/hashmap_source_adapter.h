/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Friday, April 18, 2025.
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

#ifndef MACHINA_SERVING_SERVABLES_HASHMAP_HASHMAP_SOURCE_ADAPTER_H_
#define MACHINA_SERVING_SERVABLES_HASHMAP_HASHMAP_SOURCE_ADAPTER_H_

#include <string>
#include <unordered_map>

#include "machina_serving/core/simple_loader.h"
#include "machina_serving/core/source_adapter.h"
#include "machina_serving/core/storage_path.h"
#include "machina_serving/servables/hashmap/hashmap_source_adapter.pb.h"

namespace machina {
namespace serving {

// A SourceAdapter for string-string hashmaps. It takes storage paths that give
// the locations of serialized hashmaps (in the format indicated in the config)
// and produces loaders for them.
class HashmapSourceAdapter final
    : public SimpleLoaderSourceAdapter<StoragePath,
                                       std::unordered_map<string, string>> {
 public:
  explicit HashmapSourceAdapter(const HashmapSourceAdapterConfig& config);
  ~HashmapSourceAdapter() override;

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(HashmapSourceAdapter);
};

}  // namespace serving
}  // namespace machina

#endif  // MACHINA_SERVING_SERVABLES_HASHMAP_HASHMAP_SOURCE_ADAPTER_H_
