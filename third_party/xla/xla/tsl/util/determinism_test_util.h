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
#ifndef MACHINA_XLATSL_UTIL_DETERMINISM_TEST_UTIL_H_
#define MACHINA_XLATSL_UTIL_DETERMINISM_TEST_UTIL_H_

#include "machina/xla/tsl/util/determinism.h"

namespace tsl {
namespace test {

// Enables determinism for a single test method.
class DeterministicOpsScope {
 public:
  DeterministicOpsScope() : was_enabled_(OpDeterminismRequired()) {
    EnableOpDeterminism(true);
  }
  ~DeterministicOpsScope() { EnableOpDeterminism(was_enabled_); }

 private:
  const bool was_enabled_;
};

}  // namespace test
}  // namespace tsl

#endif  // MACHINA_XLATSL_UTIL_DETERMINISM_TEST_UTIL_H_
