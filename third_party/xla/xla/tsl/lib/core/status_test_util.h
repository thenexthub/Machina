/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, March 6, 2025.
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

#ifndef MACHINA_XLATSL_LIB_CORE_STATUS_TEST_UTIL_H_
#define MACHINA_XLATSL_LIB_CORE_STATUS_TEST_UTIL_H_

#include "absl/status/status_matchers.h"  // IWYU pragma: keep
#include "machina/xla/tsl/platform/status_matchers.h"
#include "machina/xla/tsl/platform/test.h"

// Macros for testing the results of functions that return machina::Status.
#define TF_EXPECT_OK(statement) EXPECT_THAT((statement), ::absl_testing::IsOk())
#define TF_ASSERT_OK(statement) ASSERT_THAT((statement), ::absl_testing::IsOk())

// There are no EXPECT_NOT_OK/ASSERT_NOT_OK macros since they would not
// provide much value (when they fail, they would just print the OK status
// which conveys no more information than EXPECT_FALSE(status.ok());
// If you want to check for particular errors, a better alternative is with
// status matchers:
// EXPECT_THAT(s, machina::testing::StatusIs(status.code(), "message"));

#endif  // MACHINA_XLATSL_LIB_CORE_STATUS_TEST_UTIL_H_
