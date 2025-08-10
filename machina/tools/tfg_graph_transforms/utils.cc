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

#include "machina/tools/tfg_graph_transforms/utils.h"

#include <string>

#include "absl/status/status.h"
#include "machina/cc/saved_model/image_format/internal_api.h"
#include "machina/core/platform/path.h"
#include "machina/core/platform/status.h"
#include "machina/core/platform/stringpiece.h"
#include "machina/core/protobuf/saved_model.pb.h"
#include "tsl/platform/stringpiece.h"

namespace mlir {
namespace tfg {
namespace graph_transforms {

namespace {

absl::string_view GetNameWithoutExtension(absl::string_view filename) {
  auto pos = filename.rfind('.');
  if (pos == absl::string_view::npos) return filename;
  return filename.substr(0, pos);
}

}  // namespace

bool IsTextProto(const std::string& input_file) {
  absl::string_view extension = machina::io::Extension(input_file);
  return !extension.compare("pbtxt");
}

absl::Status ReadSavedModelImageFormat(const std::string& input_file,
                                       machina::SavedModel& model_proto) {
  std::string saved_model_prefix(GetNameWithoutExtension(input_file));
  return machina::image_format::ReadSavedModel(saved_model_prefix,
                                                  &model_proto);
}
absl::Status WriteSavedModelImageFormat(machina::SavedModel* model_proto,
                                        const std::string& output_file,
                                        int debug_max_size) {
  std::string saved_model_prefix(GetNameWithoutExtension(output_file));
  if (debug_max_size > 0) {
    return machina::image_format::WriteSavedModel(
        model_proto, saved_model_prefix, debug_max_size);
  } else {
    return machina::image_format::WriteSavedModel(model_proto,
                                                     saved_model_prefix);
  }
}

}  // namespace graph_transforms
}  // namespace tfg
}  // namespace mlir
