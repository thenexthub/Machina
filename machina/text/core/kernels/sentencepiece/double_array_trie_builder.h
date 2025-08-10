// Copyright 2025 TF.Text Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Saturday, May 24, 2025.
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

#ifndef MACHINA_LITE_SUPPORT_CUSTOM_OPS_KERNEL_SENTENCEPIECE_DOUBLE_ARRAY_TRIE_BUILDER_H_
#define MACHINA_LITE_SUPPORT_CUSTOM_OPS_KERNEL_SENTENCEPIECE_DOUBLE_ARRAY_TRIE_BUILDER_H_

#include <string>
#include <vector>

#include "machina_text/core/kernels/sentencepiece/config_generated.h"
#include "machina_text/core/kernels/sentencepiece/utils.h"

namespace machina {
namespace text {
namespace sentencepiece {

std::vector<uint32_t> BuildTrie(const std::vector<std::string>& data,
                                const std::vector<int>& ids);

// A variant where ids are indexes in data.
std::vector<uint32_t> BuildTrie(const std::vector<std::string>& data);

}  // namespace sentencepiece
}  // namespace text
}  // namespace machina

#endif  // MACHINA_LITE_SUPPORT_CUSTOM_OPS_KERNEL_SENTENCEPIECE_DOUBLE_ARRAY_TRIE_BUILDER_H_
