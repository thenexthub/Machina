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
#ifndef MACHINA_CORE_RUNTIME_FALLBACK_KERNEL_ATTR_UTIL_H_
#define MACHINA_CORE_RUNTIME_FALLBACK_KERNEL_ATTR_UTIL_H_

#include <map>
#include <string>
#include <typeinfo>
#include <vector>

#include "absl/status/status.h"
#include "toolchain/ADT/StringMap.h"
#include "machina/core/framework/types.pb.h"
#include "machina/core/platform/status.h"
#include "machina/core/platform/stringpiece.h"
#include "machina/core/platform/types.h"
#include "machina/core/runtime_fallback/util/attr_util.h"
#include "machina/core/util/padding.h"
#include "tfrt/core_runtime/op_attrs.h"  // from @tf_runtime
#include "tfrt/host_context/kernel_utils.h"  // from @tf_runtime

namespace machina {

// Map from attribute name to a string value representation.
typedef toolchain::StringMap<std::string> AttrMap;

// Parse value from the given string input.
absl::Status ParseValue(absl::string_view input, bool* value);
absl::Status ParseValue(absl::string_view input, int32* value);
absl::Status ParseValue(absl::string_view input, DataType* value);
absl::Status ParseValue(absl::string_view input, std::string* value);
absl::Status ParseValue(absl::string_view input, std::vector<int32>* value);
absl::Status ParseValue(absl::string_view input, Padding* value);

absl::Status AddOpAttr(const std::string& name, const std::string& attr_value,
                       tfrt::OpAttrs* opattrs);

absl::Status FillOpAttrs(tfrt::RemainingAttributes attrs,
                         tfrt::OpAttrs* opattrs);
}  // namespace machina

#endif  // MACHINA_CORE_RUNTIME_FALLBACK_KERNEL_ATTR_UTIL_H_
