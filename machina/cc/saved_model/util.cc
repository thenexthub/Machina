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
#include "machina/cc/saved_model/util.h"

#include <set>
#include <string>
#include <utility>
#include <vector>

#include "machina/core/platform/errors.h"
#include "machina/core/protobuf/meta_graph.pb.h"

namespace machina {
namespace saved_model {

std::string GetWriteVersion(const SavedModel& saved_model) {
  if (saved_model.meta_graphs_size() == 1 &&
      saved_model.meta_graphs()[0].has_object_graph_def()) {
    return "2";
  }
  return "1";
}

std::set<std::string> GetMapKeys(
    const ::google::protobuf::Map<std::string, ::machina::TensorProto>& map) {
  std::set<std::string> keys;
  for (const auto& it : map) {
    keys.insert(it.first);
  }
  return keys;
}

absl::Status GetInputValues(
    const SignatureDef& signature,
    const ::google::protobuf::Map<std::string, ::machina::TensorProto>& request_inputs,
    std::vector<std::pair<string, Tensor>>& inputs) {
  const TensorProto* tensor_proto;
  std::set<std::string> seen_request_inputs;

  for (const auto& [alias, tensor_info] : signature.inputs()) {
    const std::string& feed_name = tensor_info.name();
    auto iter = request_inputs.find(alias);

    if (iter == request_inputs.end()) {
      auto default_inputs_iter = signature.defaults().find(alias);
      if (default_inputs_iter == signature.defaults().end()) {
        return absl::InvalidArgumentError(strings::StrCat(
            "Signature input alias: ", alias, "(feed name: ", feed_name,
            ") not found in request and no default value provided. Inputs "
            "expected to be in the request {",
            absl::StrJoin(GetMapKeys(request_inputs), ","),
            "}, or default input alias set: ",
            absl::StrJoin(GetMapKeys(signature.defaults()), ",")));
      }
      tensor_proto = &default_inputs_iter->second;
    } else {
      tensor_proto = &iter->second;
      seen_request_inputs.insert(alias);
    }

    Tensor tensor;
    if (!tensor.FromProto(*tensor_proto)) {
      return absl::InvalidArgumentError(
          absl::StrCat("tensor parsing error: ", alias));
    }

    inputs.emplace_back(feed_name, tensor);
  }

  if (!request_inputs.empty() &&
      seen_request_inputs.size() != request_inputs.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Inputs contains invalid name. Used request inputs: ",
        absl::StrJoin(seen_request_inputs, ","),
        ", request input: ", absl::StrJoin(GetMapKeys(request_inputs), ",")));
  }
  return absl::OkStatus();
}

}  // namespace saved_model
}  // namespace machina
