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
#include "machina/core/tfrt/mlrt/bytecode/kernel.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "machina/core/tfrt/mlrt/bytecode/bytecode.h"

namespace mlrt {
namespace bc {
namespace {

TEST(KernelTest, Kernel) {
  Buffer buffer;
  Allocator allocator(&buffer);

  Kernel::Constructor ctor = New<Kernel>(&allocator);

  ctor.set_code(100);

  ctor.construct_arguments(/*size=*/2).Assign({400, 500});
  ctor.construct_results(/*size=*/3).Assign({100, 200, 300});
  ctor.construct_attributes(/*size=*/1).Assign({1400});
  ctor.construct_last_uses(/*size=*/2).Assign({0, 1});

  Kernel kernel(buffer.Get(ctor.address()));

  EXPECT_EQ(kernel.code(), 100);

  EXPECT_THAT(kernel.arguments(), testing::ElementsAreArray({400, 500}));
  EXPECT_THAT(kernel.results(), testing::ElementsAreArray({100, 200, 300}));
  EXPECT_THAT(kernel.attributes(), testing::ElementsAreArray({1400}));
  EXPECT_THAT(kernel.last_uses(), testing::ElementsAreArray({0, 1}));
}

}  // namespace
}  // namespace bc
}  // namespace mlrt
