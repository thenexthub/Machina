/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Friday, August 8, 2025.
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

#ifndef MACHINA_LITE_EXPERIMENTAL_SHLO_TEST_MACROS_H_
#define MACHINA_LITE_EXPERIMENTAL_SHLO_TEST_MACROS_H_

// IWYU pragma: always_keep

#include <gmock/gmock.h>
#include "absl/status/status.h"  // IWYU pragma: keep - used in the
                                             // provided macros in OSS builds.

namespace shlo_ref {
namespace testing {

MATCHER_P(StatusIs, status_code, "") { return arg.code() == status_code; }

#ifndef ASSERT_OK
#define ASSERT_OK(x) \
  ASSERT_THAT(x, ::shlo_ref::testing::StatusIs(::absl::StatusCode::kOk))
#endif

#ifndef EXPECT_OK
#define EXPECT_OK(x) \
  EXPECT_THAT(x, ::shlo_ref::testing::StatusIs(::absl::StatusCode::kOk))
#endif

}  // namespace testing
}  // namespace shlo_ref

#endif  // MACHINA_LITE_EXPERIMENTAL_SHLO_TEST_MACROS_H_
