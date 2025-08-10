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
#if GOOGLE_CUDA && GOOGLE_TENSORRT
#include "machina/compiler/tf2tensorrt/convert/op_converter_registry.h"

#include <gtest/gtest.h>
#include "machina/compiler/tf2tensorrt/convert/op_converter.h"

namespace machina {
namespace tensorrt {
namespace convert {

TEST(TestOpConverterRegistry, TestOpConverterRegistry) {
  bool flag{false};

  auto set_true_func = [&flag](const OpConverterParams*) -> Status {
    flag = true;
    return OkStatus();
  };

  auto set_false_func = [&flag](const OpConverterParams*) -> Status {
    flag = false;
    return OkStatus();
  };

  GetOpConverterRegistry()->Register("FakeFunc", kDefaultConverterPriority,
                                     set_true_func);

  // Lower priority fails to override.
  GetOpConverterRegistry()->Register("FakeFunc", kDefaultConverterPriority - 1,
                                     set_false_func);

  // The lookup should return set_true_func (default).
  auto func = GetOpConverterRegistry()->LookUp("FakeFunc");
  EXPECT_TRUE(func.ok());
  EXPECT_TRUE(((*func)(nullptr)).ok());
  EXPECT_TRUE(flag);

  // Override with higher priority.
  GetOpConverterRegistry()->Register("FakeFunc", kDefaultConverterPriority + 1,
                                     set_false_func);
  func = GetOpConverterRegistry()->LookUp("FakeFunc");
  EXPECT_TRUE(func.ok());
  EXPECT_TRUE((*func)(nullptr).ok());
  EXPECT_FALSE(flag);

  // After clearing the op, lookup should return an error.
  GetOpConverterRegistry()->Clear("FakeFunc");
  EXPECT_FALSE(GetOpConverterRegistry()->LookUp("FakeFunc").ok());
}
}  // namespace convert
}  // namespace tensorrt
}  // namespace machina

#endif
