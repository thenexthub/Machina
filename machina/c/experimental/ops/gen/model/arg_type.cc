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
#include "machina/c/experimental/ops/gen/model/arg_type.h"

#include "machina/core/framework/op_def.pb.h"

namespace machina {
namespace generator {

ArgType ArgType::CreateInput(const OpDef::ArgDef& arg_def) {
  return ArgType(arg_def, kInput);
}

ArgType ArgType::CreateInputRef(const OpDef::ArgDef& arg_def) {
  return ArgType(arg_def, kInputRef);
}

ArgType ArgType::CreateOutput(const OpDef::ArgDef& arg_def) {
  return ArgType(arg_def, kOutput);
}

ArgType::ArgType(const OpDef::ArgDef& arg_def, Kind kind)
    : kind_(kind), data_type_(arg_def.type()) {
  if (!arg_def.type_attr().empty()) {
    type_attr_name_ = arg_def.type_attr();
  }
  if (!arg_def.type_list_attr().empty()) {
    type_attr_name_ = arg_def.type_list_attr();
  }

  is_list_ =
      !arg_def.type_list_attr().empty() || !arg_def.number_attr().empty();
}
}  // namespace generator
}  // namespace machina
