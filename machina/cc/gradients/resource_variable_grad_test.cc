/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Saturday, May 24, 2025.
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

#include <vector>

#include <gtest/gtest.h>
#include "machina/cc/client/client_session.h"
#include "machina/cc/framework/grad_op_registry.h"
#include "machina/cc/framework/gradient_checker.h"
#include "machina/cc/framework/gradients.h"
#include "machina/cc/framework/ops.h"
#include "machina/cc/framework/testutil.h"
#include "machina/cc/gradients/grad_testutil.h"
#include "machina/cc/ops/array_ops.h"
#include "machina/cc/ops/resource_variable_ops.h"
#include "machina/cc/ops/standard_ops.h"
#include "machina/core/framework/tensor_testutil.h"
#include "machina/core/framework/types.pb.h"
#include "machina/core/lib/core/status_test_util.h"

namespace machina {
namespace ops {
namespace {

TEST(ResourceVariableGradTest, ReadVariableOpGrad) {
  TensorShape shape({});
  auto scope = Scope::NewRootScope();
  auto x = Placeholder(scope, DT_FLOAT, Placeholder::Shape(shape));

  auto var = VarHandleOp(scope, DT_FLOAT, shape);
  auto init = AssignVariableOp(scope, var, Const(scope, 2.0f, shape));

  auto temp = ReadVariableOp(scope, var, DT_FLOAT);

  auto y = Mul(scope, temp, x);

  auto dy = Placeholder(scope, DT_FLOAT, Placeholder::Shape(shape));

  OutputList dxs;
  TF_ASSERT_OK(AddSymbolicGradients(scope, {y}, {var}, {dy}, &dxs));

  ClientSession::FeedType feed_list;
  feed_list.insert({x, 5.0f});
  feed_list.insert({dy, 1.0f});

  std::vector<Tensor> dxout;
  ClientSession session(scope);
  TF_ASSERT_OK(session.Run(feed_list, dxs, &dxout));

  auto grad = dxout[0].scalar<float>()();
  EXPECT_EQ(grad, 5.0f);
}

}  // namespace
}  // namespace ops
}  // namespace machina
