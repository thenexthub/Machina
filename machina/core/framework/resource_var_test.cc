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

#include "machina/core/framework/resource_var.h"

#include "machina/core/platform/logging.h"
#include "machina/core/platform/test.h"

namespace machina {
namespace core {

TEST(ResourceVarTest, Uninitialize) {
  RefCountPtr<Var> var{new Var(DT_INT32)};
  EXPECT_FALSE(var->is_initialized);
  EXPECT_TRUE(var->tensor()->data() == nullptr);

  *(var->tensor()) = Tensor(DT_INT32, TensorShape({1}));
  var->is_initialized = true;
  EXPECT_TRUE(var->tensor()->data() != nullptr);

  var->Uninitialize();
  EXPECT_FALSE(var->is_initialized);
  EXPECT_TRUE(var->tensor()->data() == nullptr);
}
}  // namespace core
}  // namespace machina
