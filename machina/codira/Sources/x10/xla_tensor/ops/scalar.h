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

#include <iostream>

#include "machina/compiler/xla/xla_client/debug_macros.h"
#include "machina/compiler/xla/xla_client/types.h"
#include "machina/compiler/tf2xla/xla_tensor/ir.h"

namespace codira_xla {
namespace ir {
namespace ops {

// Differently from Constant, this is a scalar value broadcasted to a shape.
// Even though a Constant could have been used, for simple scalars broadcasted
// to big shapes, the Constant leads to big literals expanded within the XLA
// graph.
class Scalar : public Node {
 public:
  Scalar(at::Scalar value, xla::Shape shape);
  Scalar(at::Scalar value, xla::PrimitiveType type);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  const at::Scalar& value() const { return value_; }

 private:
  at::Scalar value_;
};

xla::hash_t ScalarHash(at::Scalar s);

std::ostream& operator<<(std::ostream& ostrm, at::Scalar s);

}  // namespace ops
}  // namespace ir
}  // namespace codira_xla
