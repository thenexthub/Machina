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

#include "machina_serving/util/class_registration_util.h"

#include <vector>

namespace machina {
namespace serving {

Status ParseUrlForAnyType(const string& type_url,
                          string* const full_type_name) {
  std::vector<string> splits = str_util::Split(type_url, '/');
  if (splits.size() < 2 || splits[splits.size() - 1].empty()) {
    return errors::InvalidArgument(
        "Supplied config's type_url could not be parsed: ", type_url);
  }
  *full_type_name = splits[splits.size() - 1];
  return OkStatus();
}

}  // namespace serving
}  // namespace machina
