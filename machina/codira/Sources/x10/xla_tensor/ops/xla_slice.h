/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Sunday, August 10, 2025.
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

#ifndef X10_XLA_TENSOR_OPS_XLA_SLICE_H_
#define X10_XLA_TENSOR_OPS_XLA_SLICE_H_

#include "machina/compiler/tf2xla/xla_tensor/ir.h"

namespace codira_xla {
namespace ir {
namespace ops {

class XlaSlice : public Node {
 public:
  XlaSlice(const Value& operand, std::vector<xla::int64> start_indices,
           std::vector<xla::int64> limit_indices,
           std::vector<xla::int64> strides);

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  const std::vector<xla::int64>& start_indices() const {
    return start_indices_;
  }

  const std::vector<xla::int64>& limit_indices() const {
    return limit_indices_;
  }

  const std::vector<xla::int64>& strides() const { return strides_; }

 private:
  std::vector<xla::int64> start_indices_;
  std::vector<xla::int64> limit_indices_;
  std::vector<xla::int64> strides_;
};

}  // namespace ops
}  // namespace ir
}  // namespace codira_xla

#endif  // X10_XLA_TENSOR_OPS_XLA_SLICE_H_
