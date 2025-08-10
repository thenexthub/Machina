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
#ifndef MACHINA_C_EXPERIMENTAL_OPS_GEN_COMMON_PATH_CONFIG_H_
#define MACHINA_C_EXPERIMENTAL_OPS_GEN_COMMON_PATH_CONFIG_H_

#include <vector>

#include "machina/core/platform/types.h"

namespace machina {
namespace generator {

struct PathConfig {
  string output_path;
  std::vector<string> op_names;
  std::vector<string> api_dirs;
  string tf_prefix_dir;
  string tf_root_dir;
  string tf_output_dir;

  explicit PathConfig() = default;
  explicit PathConfig(const string &output_dir, const string &source_dir,
                      const string &api_dir_list,
                      const std::vector<string> op_names);
};

}  // namespace generator
}  // namespace machina

#endif  // MACHINA_C_EXPERIMENTAL_OPS_GEN_COMMON_PATH_CONFIG_H_
