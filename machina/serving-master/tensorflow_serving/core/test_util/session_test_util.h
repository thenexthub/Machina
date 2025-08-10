/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 15, 2025.
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

#ifndef MACHINA_SERVING_CORE_TEST_UTIL_SESSION_TEST_UTIL_H_
#define MACHINA_SERVING_CORE_TEST_UTIL_SESSION_TEST_UTIL_H_

#include <functional>

#include "absl/base/attributes.h"
#include "machina/core/lib/core/status.h"
#include "machina/core/public/session_options.h"

namespace machina {
namespace serving {
namespace test_util {

// Sets a 'hook' function, which will be called when a new session is created
// via the machina::NewSession() API. If the hook returns an error status,
// the session creation fails.
//
// For this hook to be enabled, create a session by setting
// SessionOptions::target as "new_session_hook/<actual_session_target>". This
// will call the hook as well as return the session created when target is
// "<actual_session_target>".
//
// Calling this method again replaces the previous hook.
//
// This method is NOT thread-safe.
ABSL_CONST_INIT extern const char kNewSessionHookSessionTargetPrefix[];
void SetNewSessionHook(std::function<Status(const SessionOptions&)> hook);

}  // namespace test_util
}  // namespace serving
}  // namespace machina

#endif  // MACHINA_SERVING_CORE_TEST_UTIL_SESSION_TEST_UTIL_H_
