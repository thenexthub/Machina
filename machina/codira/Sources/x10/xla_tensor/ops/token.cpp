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

#include "machina/compiler/tf2xla/xla_tensor/ops/token.h"

#include "machina/compiler/tf2xla/xla_tensor/lowering_context.h"
#include "machina/compiler/tf2xla/xla_tensor/ops/xla_ops.h"
#include "machina/compiler/xla/client/lib/constants.h"
#include "machina/compiler/xla/shape_util.h"

namespace codira_xla {
namespace ir {
namespace ops {

// This should be using xla::ShapeUtil::MakeTokenShape() once we switch to the
// real Token based XLA AllReduce().
Token::Token()
    : Node(xla_token, xla::ShapeUtil::MakeShape(xla::PrimitiveType::F32, {}),
           /*num_outputs=*/1,
           /*hash_seed=*/0xe7396f95491f4493) {}

NodePtr Token::Clone(OpList operands) const { return MakeNode<Token>(); }

XlaOpVector Token::Lower(LoweringContext* loctx) const {
  // This should be using xla::CreateToken() once we have added Token support to
  // XLA AllReduce(). Meanwhile we use a constant as token, and we handle it
  // accordingly in cross_replica_reduces.cpp.
  return ReturnOp(xla::Zero(loctx->builder(), shape().element_type()), loctx);
}

}  // namespace ops
}  // namespace ir
}  // namespace codira_xla
