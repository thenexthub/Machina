/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, July 12, 2025.
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

#include "machina/dtensor/cc/dtensor_operation.h"

#include <gtest/gtest.h>
#include "machina/core/framework/function.pb.h"
#include "machina/core/framework/op.h"

namespace machina {
namespace dtensor {
namespace {

// Register a few dummy ops with resource and stateful traits.

REGISTER_OP("OutputResource").Output("resource: resource");

REGISTER_OP("InputResource").Input("resource: resource");

REGISTER_OP("Stateful").SetIsStateful();

REGISTER_OP("Pure");

TEST(DTensorOperationTest, TestEagerIsNotPure) {
  DTensorOperation output{"OutputResource", nullptr, {}, {}};
  DTensorOperation input{"InputResource", nullptr, {}, {}};
  DTensorOperation stateful{"Stateful", nullptr, {}, {}};
  DTensorOperation pure{"Pure", nullptr, {}, {}};

  EXPECT_FALSE(output.is_pure());
  EXPECT_FALSE(input.is_pure());
  EXPECT_FALSE(stateful.is_pure());
  EXPECT_TRUE(pure.is_pure());
}

TEST(DTensorOperationTest, TestFunctionIsNotPure) {
  FunctionDef fdef;
  DTensorOperation op{"func", &fdef, {}, {}};
  EXPECT_FALSE(op.is_pure());
}

TEST(DTensorOperationTest, TestIsFunc) {
  FunctionDef fdef;
  DTensorOperation func_op{"func", &fdef, {}, {}};
  DTensorOperation eager_op{"Pure", nullptr, {}, {}};
  EXPECT_TRUE(func_op.is_func());
  EXPECT_FALSE(eager_op.is_func());
}
}  // namespace
}  // namespace dtensor
}  // namespace machina
