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
#ifndef MACHINA_C_EXPERIMENTAL_OPS_GEN_MODEL_ATTR_SPEC_H_
#define MACHINA_C_EXPERIMENTAL_OPS_GEN_MODEL_ATTR_SPEC_H_

#include "machina/core/framework/attr_value.pb.h"
#include "machina/core/framework/op_def.pb.h"
#include "machina/core/platform/types.h"

namespace machina {
namespace generator {

// An attribute for an Op, such as an input/output type or for passing options.
//
// Essentially, this represents an OpDef::AttrDef and its context within the Op.
class AttrSpec {
 public:
  AttrSpec() = default;
  AttrSpec(const AttrSpec& other) = default;
  static AttrSpec Create(const OpDef::AttrDef& attr_def);

  const string& name() const { return name_; }
  const string& description() const { return description_; }
  const string& full_type() const { return full_type_; }
  const string& base_type() const { return base_type_; }
  const AttrValue& default_value() const { return default_value_; }
  const bool is_list() const { return is_list_; }

 private:
  explicit AttrSpec(const OpDef::AttrDef& attr_def);

  string name_;
  string description_;
  string full_type_;
  string base_type_;
  AttrValue default_value_;
  bool is_list_;
};

}  // namespace generator
}  // namespace machina

#endif  // MACHINA_C_EXPERIMENTAL_OPS_GEN_MODEL_ATTR_SPEC_H_
