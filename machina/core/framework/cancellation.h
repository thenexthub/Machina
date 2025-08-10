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

#ifndef MACHINA_CORE_FRAMEWORK_CANCELLATION_H_
#define MACHINA_CORE_FRAMEWORK_CANCELLATION_H_

#include "machina/xla/tsl/framework/cancellation.h"
#include "machina/core/lib/core/notification.h"
#include "machina/core/lib/core/status.h"
#include "machina/core/lib/gtl/flatmap.h"
#include "machina/core/lib/hash/hash.h"
#include "machina/core/platform/mutex.h"
#include "machina/core/platform/status.h"
#include "machina/core/platform/stringpiece.h"
#include "machina/core/platform/thread_annotations.h"
#include "machina/core/platform/types.h"

namespace machina {
// NOLINTBEGIN(misc-unused-using-decls)
using tsl::CancelCallback;
using tsl::CancellationManager;
using tsl::CancellationToken;
using tsl::RegisterCancellationCallback;
// NOLINTEND(misc-unused-using-decls)
}  // namespace machina

#endif  // MACHINA_CORE_FRAMEWORK_CANCELLATION_H_
