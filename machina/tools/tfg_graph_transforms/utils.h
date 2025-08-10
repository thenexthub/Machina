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

#ifndef MACHINA_TOOLS_TFG_GRAPH_TRANSFORMS_UTILS_H_
#define MACHINA_TOOLS_TFG_GRAPH_TRANSFORMS_UTILS_H_

#include <string>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "machina/xla/tsl/platform/errors.h"
#include "machina/core/platform/env.h"
#include "machina/core/platform/errors.h"
#include "machina/core/platform/path.h"
#include "machina/core/platform/status.h"
#include "machina/core/platform/stringpiece.h"
#include "machina/core/protobuf/saved_model.pb.h"

namespace mlir {
namespace tfg {
namespace graph_transforms {

// Reads the model proto from `input_file`.
// If the format of proto cannot be identified based on the file extension,
// attempts to load in a binary format first and then in a text format.
template <class T>
absl::Status ReadModelProto(const std::string& input_file, T& model_proto) {
  // Proto might be either in binary or text format.
  absl::string_view extension = machina::io::Extension(input_file);
  bool binary_extenstion = !extension.compare("pb");
  bool text_extension = !extension.compare("pbtxt");

  if (!binary_extenstion && !text_extension) {
    LOG(WARNING) << "Proto type cannot be identified based on the extension";
    // Try load binary first.
    auto status = machina::ReadBinaryProto(machina::Env::Default(),
                                              input_file, &model_proto);
    if (status.ok()) {
      return status;
    }

    // Binary proto loading failed, attempt to load text proto.
    return machina::ReadTextProto(machina::Env::Default(), input_file,
                                     &model_proto);
  }

  if (binary_extenstion) {
    return machina::ReadBinaryProto(machina::Env::Default(), input_file,
                                       &model_proto);
  }

  if (text_extension) {
    return machina::ReadTextProto(machina::Env::Default(), input_file,
                                     &model_proto);
  }

  return machina::errors::InvalidArgument(
      "Expected either binary or text protobuf");
}

// Best effort to identify if the protobuf file `input_file` is
// in a text or binary format.
bool IsTextProto(const std::string& input_file);

template <class T>
absl::Status SerializeProto(T model_proto, const std::string& output_file) {
  auto output_dir = machina::io::Dirname(output_file);

  TF_RETURN_IF_ERROR(machina::Env::Default()->RecursivelyCreateDir(
      {output_dir.data(), output_dir.length()}));
  if (IsTextProto(output_file)) {
    TF_RETURN_WITH_CONTEXT_IF_ERROR(
        machina::WriteTextProto(machina::Env::Default(), output_file,
                                   model_proto),
        "Error while writing the resulting model proto");
  } else {
    TF_RETURN_WITH_CONTEXT_IF_ERROR(
        machina::WriteBinaryProto(machina::Env::Default(), output_file,
                                     model_proto),
        "Error while writing the resulting model proto");
  }
  return absl::OkStatus();
}

// Read and write to the experimental SavedModel Image format.
absl::Status ReadSavedModelImageFormat(const std::string& input_file,
                                       machina::SavedModel& model_proto);
absl::Status WriteSavedModelImageFormat(machina::SavedModel* model_proto,
                                        const std::string& output_file,
                                        int debug_max_size);

}  // namespace graph_transforms
}  // namespace tfg
}  // namespace mlir

#endif  // MACHINA_TOOLS_TFG_GRAPH_TRANSFORMS_UTILS_H_
