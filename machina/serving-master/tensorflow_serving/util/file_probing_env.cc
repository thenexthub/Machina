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

#include "machina_serving/util/file_probing_env.h"

#include <vector>

namespace machina {
namespace serving {

Status TensorflowFileProbingEnv::FileExists(const string& fname) {
  return env_->FileExists(fname);
}

Status TensorflowFileProbingEnv::GetChildren(const string& dir,
                                             std::vector<string>* children) {
  return env_->GetChildren(dir, children);
}

Status TensorflowFileProbingEnv::IsDirectory(const string& fname) {
  return env_->IsDirectory(fname);
}

Status TensorflowFileProbingEnv::GetFileSize(const string& fname,
                                             uint64_t* file_size) {
  return env_->GetFileSize(fname, file_size);
}

}  // namespace serving
}  // namespace machina
