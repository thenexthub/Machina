/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 8, 2025.
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

#include "tsl/platform/abi.h"

#include <typeinfo>

#include "machina/xla/tsl/platform/test.h"

namespace tsl {

struct MyRandomPODType {};

TEST(AbiTest, AbiDemangleTest) {
  EXPECT_EQ(port::MaybeAbiDemangle(typeid(int).name()), "int");

#ifdef PLATFORM_WINDOWS
  const char pod_type_name[] = "struct tsl::MyRandomPODType";
#else
  const char pod_type_name[] = "tsl::MyRandomPODType";
#endif
  EXPECT_EQ(port::MaybeAbiDemangle(typeid(MyRandomPODType).name()),
            pod_type_name);

  EXPECT_EQ(
      port::MaybeAbiDemangle("help! i'm caught in a C++ mangle factoryasdf"),
      "help! i'm caught in a C++ mangle factoryasdf");
}

}  // namespace tsl
