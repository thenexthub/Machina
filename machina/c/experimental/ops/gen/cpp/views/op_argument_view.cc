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
#include "machina/c/experimental/ops/gen/cpp/views/op_argument_view.h"

#include "absl/strings/substitute.h"
#include "machina/c/experimental/ops/gen/cpp/views/arg_type_view.h"
#include "machina/c/experimental/ops/gen/cpp/views/arg_view.h"
#include "machina/c/experimental/ops/gen/cpp/views/attr_view.h"
#include "machina/c/experimental/ops/gen/model/arg_spec.h"
#include "machina/c/experimental/ops/gen/model/attr_spec.h"
#include "machina/core/platform/types.h"

namespace machina {
namespace generator {
namespace cpp {

string OpArgumentView::Declaration() const {
  return absl::Substitute("$0 $1", type_name_, variable_name_);
}

string OpArgumentView::Initializer() const {
  if (default_value_.empty()) {
    return "";
  }
  return absl::Substitute(" = $0", default_value_);
}

bool OpArgumentView::HasDefaultValue() const { return !default_value_.empty(); }

OpArgumentView::OpArgumentView(string type, string var, string def)
    : type_name_(type), variable_name_(var), default_value_(def) {}

OpArgumentView::OpArgumentView(ArgSpec arg)
    : type_name_(ArgTypeView(arg.arg_type()).TypeName()),
      variable_name_(ArgView(arg).VariableName()) {}

OpArgumentView::OpArgumentView(AttrSpec attr)
    : type_name_(AttrView(attr).VariableType()),
      variable_name_(AttrView(attr).VariableName()),
      default_value_(AttrView(attr).DefaultValue()) {}

}  // namespace cpp
}  // namespace generator
}  // namespace machina
