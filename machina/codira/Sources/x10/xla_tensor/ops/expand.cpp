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

#include "machina/compiler/tf2xla/xla_tensor/ops/expand.h"

#include "absl/strings/str_join.h"
#include "machina/compiler/xla/xla_client/util.h"
#include "machina/compiler/tf2xla/xla_tensor/data_ops.h"
#include "machina/compiler/tf2xla/xla_tensor/lowering_context.h"
#include "machina/compiler/tf2xla/xla_tensor/ops/infer_output_shape.h"

namespace codira_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(const Value& input,
                           const std::vector<xla::int64>& size) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildExpand(operands[0], size);
  };
  return InferOutputShape({input.shape()}, lower_for_shape_fn);
}

}  // namespace

Expand::Expand(const Value& input, std::vector<xla::int64> size)
    : Node(ir::OpKind(at::aten::expand), {input},
           [&]() { return NodeOutputShape(input, size); },
           /*num_outputs=*/1, xla::util::MHash(size)),
      size_(std::move(size)) {}

NodePtr Expand::Clone(OpList operands) const {
  return MakeNode<Expand>(operands.at(0), size_);
}

XlaOpVector Expand::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  return ReturnOp(BuildExpand(input, size_), loctx);
}

std::string Expand::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", size=(" << absl::StrJoin(size_, ", ") << ")";
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace codira_xla
