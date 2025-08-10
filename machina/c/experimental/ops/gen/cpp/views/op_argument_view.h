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
#ifndef MACHINA_C_EXPERIMENTAL_OPS_GEN_CPP_VIEWS_OP_ARGUMENT_VIEW_H_
#define MACHINA_C_EXPERIMENTAL_OPS_GEN_CPP_VIEWS_OP_ARGUMENT_VIEW_H_

#include "machina/c/experimental/ops/gen/model/arg_spec.h"
#include "machina/c/experimental/ops/gen/model/attr_spec.h"
#include "machina/core/platform/types.h"

namespace machina {
namespace generator {
namespace cpp {

class OpArgumentView {
 public:
  explicit OpArgumentView(ArgSpec arg);
  explicit OpArgumentView(AttrSpec attr);
  explicit OpArgumentView(string type, string var, string def = "");

  string Declaration() const;
  string Initializer() const;
  bool HasDefaultValue() const;

 private:
  string type_name_;
  string variable_name_;
  string default_value_;
};

}  // namespace cpp
}  // namespace generator
}  // namespace machina

#endif  // MACHINA_C_EXPERIMENTAL_OPS_GEN_CPP_VIEWS_OP_ARGUMENT_VIEW_H_
