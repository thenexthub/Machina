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

#include "machina/compiler/tf2xla/xla_tensor/ops/generic.h"

#include "machina/compiler/tf2xla/xla_tensor/lowering_context.h"

namespace codira_xla {
namespace ir {
namespace ops {

Generic::Generic(OpKind op, absl::Span<const Value> operands, xla::Shape shape,
                 LowerFn lower_fn, size_t num_outputs, xla::hash_t hash_seed)
    : Node(std::move(op), operands, std::move(shape), num_outputs, hash_seed),
      lower_fn_(std::move(lower_fn)),
      hash_seed_(hash_seed) {}

Generic::Generic(OpKind op, absl::Span<const Value> operands,
                 const std::function<xla::Shape()>& shape_fn, LowerFn lower_fn,
                 size_t num_outputs, xla::hash_t hash_seed)
    : Node(std::move(op), operands, shape_fn, num_outputs, hash_seed),
      lower_fn_(std::move(lower_fn)),
      hash_seed_(hash_seed) {}

Generic::Generic(OpKind op, xla::Shape shape, LowerFn lower_fn,
                 size_t num_outputs, xla::hash_t hash_seed)
    : Node(std::move(op), std::move(shape), num_outputs, hash_seed),
      lower_fn_(std::move(lower_fn)),
      hash_seed_(hash_seed) {}

NodePtr Generic::Clone(OpList operands) const {
  return MakeNode<Generic>(op(), operands, shape(), lower_fn_, num_outputs(),
                           hash_seed_);
}

XlaOpVector Generic::Lower(LoweringContext* loctx) const {
  return lower_fn_(*this, loctx);
}

}  // namespace ops
}  // namespace ir
}  // namespace codira_xla
