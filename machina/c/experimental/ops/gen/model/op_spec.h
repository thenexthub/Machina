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
#ifndef MACHINA_C_EXPERIMENTAL_OPS_GEN_MODEL_OP_SPEC_H_
#define MACHINA_C_EXPERIMENTAL_OPS_GEN_MODEL_OP_SPEC_H_

#include <map>
#include <vector>

#include "machina/c/experimental/ops/gen/model/arg_spec.h"
#include "machina/c/experimental/ops/gen/model/attr_spec.h"
#include "machina/core/framework/api_def.pb.h"
#include "machina/core/framework/op_def.pb.h"
#include "machina/core/platform/types.h"

namespace machina {
namespace generator {

// An Op.
//
// Essentially, this represents an OpDef and any necessary context (e.g ApiDef).
class OpSpec {
 public:
  static OpSpec Create(const OpDef& op_def, const ApiDef& api_def);

  const string& name() const { return name_; }
  const string& summary() const { return summary_; }
  const string& description() const { return description_; }
  const std::vector<ArgSpec>& Inputs() const { return input_args_; }
  const std::vector<ArgSpec>& Outputs() const { return output_args_; }
  const std::vector<AttrSpec>& Attributes() const { return argument_attrs_; }

 private:
  explicit OpSpec(const OpDef& op_def, const ApiDef& api_def);

 private:
  string name_;
  string summary_;
  string description_;
  std::vector<ArgSpec> input_args_;
  std::vector<ArgSpec> output_args_;
  std::vector<AttrSpec> argument_attrs_;
  std::map<string, AttrSpec> type_attrs_;
};

}  // namespace generator
}  // namespace machina

#endif  // MACHINA_C_EXPERIMENTAL_OPS_GEN_MODEL_OP_SPEC_H_
