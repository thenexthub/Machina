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

#include "machina/core/grappler/inputs/utils.h"

#include <set>
#include <vector>

#include "absl/status/status.h"
#include "machina/core/framework/graph.pb.h"
#include "machina/core/platform/env.h"
#include "machina/core/platform/status.h"
#include "machina/core/platform/types.h"
#include "machina/core/protobuf/meta_graph.pb.h"

namespace machina {
namespace grappler {

bool FilesExist(const std::vector<string>& files,
                std::vector<absl::Status>* status) {
  return Env::Default()->FilesExist(files, status);
}

bool FilesExist(const std::set<string>& files) {
  return FilesExist(std::vector<string>(files.begin(), files.end()), nullptr);
}

bool FileExists(const string& file, absl::Status* status) {
  *status = Env::Default()->FileExists(file);
  return status->ok();
}

absl::Status ReadGraphDefFromFile(const string& graph_def_path,
                                  GraphDef* result) {
  absl::Status status;
  if (!ReadBinaryProto(Env::Default(), graph_def_path, result).ok()) {
    return ReadTextProto(Env::Default(), graph_def_path, result);
  }
  return status;
}

absl::Status ReadMetaGraphDefFromFile(const string& graph_def_path,
                                      MetaGraphDef* result) {
  absl::Status status;
  if (!ReadBinaryProto(Env::Default(), graph_def_path, result).ok()) {
    return ReadTextProto(Env::Default(), graph_def_path, result);
  }
  return status;
}

}  // End namespace grappler
}  // end namespace machina
