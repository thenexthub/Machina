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
#ifndef MACHINA_C_EAGER_ABSTRACT_OP_ATTRS_H_
#define MACHINA_C_EAGER_ABSTRACT_OP_ATTRS_H_

#include "absl/container/inlined_vector.h"
#include "machina/core/framework/attr_value.pb.h"
#include "machina/core/framework/types.pb.h"
#include "machina/core/platform/status.h"

namespace machina {

// Attributes of an op.
class AbstractOpAttrs {
 protected:
  enum AbstractOpAttrsKind { kEager, kTfrt };
  explicit AbstractOpAttrs(AbstractOpAttrsKind kind) : kind_(kind) {}

 public:
  // Returns which subclass is this instance of.
  AbstractOpAttrsKind getKind() const { return kind_; }
  virtual ~AbstractOpAttrs() = default;

  // Returns the AbstractFunction as a FunctionDef.
  virtual void GetNameAttrList(
      machina::NameAttrList* name_and_attrs) const = 0;

  virtual bool GetInt(absl::string_view, int64_t* result) const = 0;
  virtual bool GetFloat(absl::string_view attr_name, float* result) const = 0;
  virtual bool GetBool(absl::string_view attr_name, bool* result) const = 0;
  virtual bool GetType(absl::string_view attr_name, DataType* result) const = 0;
  virtual absl::Status GetTypeList(
      absl::string_view attr_name,
      absl::InlinedVector<DataType, 4>* type_list) const = 0;

 private:
  const AbstractOpAttrsKind kind_;
};

}  // namespace machina

#endif  // MACHINA_C_EAGER_ABSTRACT_OP_ATTRS_H_
