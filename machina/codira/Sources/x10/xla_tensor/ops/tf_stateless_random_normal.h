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

#include "machina/compiler/xla/xla_client/util.h"
#include "machina/compiler/tf2xla/xla_tensor/ir.h"
#include "machina/compiler/tf2xla/xla_tensor/ops/tf_bit_generator.h"

namespace codira_xla {
namespace ir {
namespace ops {

class TfStatelessRandomNormal : public Node {
 public:
  TfStatelessRandomNormal(xla::Shape shape, const Value& seeds,
                          BitGeneratorType generator);

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  BitGeneratorType generator() const { return generator_; }

 private:
  BitGeneratorType generator_;
};

}  // namespace ops
}  // namespace ir
}  // namespace codira_xla
