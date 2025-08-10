/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 26, 2025.
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

#ifndef MACHINA_TSL_PLATFORM_NOTIFICATION_H_
#define MACHINA_TSL_PLATFORM_NOTIFICATION_H_

#include <chrono>              // NOLINT
#include <condition_variable>  // NOLINT
#include <cstdint>
#include <mutex>  // NOLINT

#include "absl/base/macros.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"

namespace tsl {

using Notification ABSL_DEPRECATE_AND_INLINE() = absl::Notification;

}  // namespace tsl

#endif  // MACHINA_TSL_PLATFORM_NOTIFICATION_H_
