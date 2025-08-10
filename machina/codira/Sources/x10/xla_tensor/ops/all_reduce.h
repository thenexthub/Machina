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

#pragma once

#include "machina/compiler/tf2xla/xla_tensor/cross_replica_reduces.h"
#include "machina/compiler/tf2xla/xla_tensor/ir.h"

namespace codira_xla {
namespace ir {
namespace ops {

class AllReduce : public Node {
 public:
  AllReduce(AllReduceType reduce_type, absl::Span<const Value> operands,
            const Value& token, double scale,
            std::vector<std::vector<xla::int64>> groups);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  AllReduceType reduce_type() const { return reduce_type_; }

  double scale() const { return scale_; }

  const std::vector<std::vector<xla::int64>>& groups() const { return groups_; }

 private:
  AllReduceType reduce_type_;
  double scale_;
  std::vector<std::vector<xla::int64>> groups_;
};

}  // namespace ops
}  // namespace ir
}  // namespace codira_xla
