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
#include "machina/lite/tools/gen_op_registration.h"

#include <algorithm>
#include <string>
#include <vector>

#include "re2/re2.h"
#include "machina/lite/core/model.h"
#include "machina/lite/schema/schema_utils.h"

namespace tflite {

string NormalizeCustomOpName(const string& op) {
  string method(op);
  RE2::GlobalReplace(&method, "([a-z])([A-Z])", "\\1_\\2");
  std::transform(method.begin(), method.end(), method.begin(), ::toupper);
  return method;
}

void ReadOpsFromModel(const ::tflite::Model* model,
                      tflite::RegisteredOpMap* builtin_ops,
                      tflite::RegisteredOpMap* custom_ops) {
  if (!model) return;
  auto opcodes = model->operator_codes();
  if (!opcodes) return;
  for (const auto* opcode : *opcodes) {
    const int version = opcode->version();
    auto builtin_code = GetBuiltinCode(opcode);
    if (builtin_code != ::tflite::BuiltinOperator_CUSTOM) {
      auto iter_and_bool = builtin_ops->insert(
          std::make_pair(tflite::EnumNameBuiltinOperator(builtin_code),
                         std::make_pair(version, version)));
      auto& versions = iter_and_bool.first->second;
      versions.first = std::min(versions.first, version);
      versions.second = std::max(versions.second, version);
    } else {
      auto iter_and_bool = custom_ops->insert(std::make_pair(
          opcode->custom_code()->c_str(), std::make_pair(version, version)));
      auto& versions = iter_and_bool.first->second;
      versions.first = std::min(versions.first, version);
      versions.second = std::max(versions.second, version);
    }
  }
}

}  // namespace tflite
