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

#ifndef MACHINA_CORE_PLATFORM_STATUS_H_
#define MACHINA_CORE_PLATFORM_STATUS_H_

#include "absl/base/macros.h"
#include "absl/status/status.h"
#include "machina/core/platform/logging.h"
#include "machina/core/platform/macros.h"
#include "machina/core/platform/stack_frame.h"
#include "machina/core/platform/types.h"
#include "tsl/platform/status.h"

namespace machina {
// NOLINTBEGIN(misc-unused-using-decls)
#ifdef SWIG
using tsl::FromAbslStatus;
using tsl::OkStatus;
using tsl::Status;
using tsl::ToAbslStatus;
#else
ABSL_DEPRECATE_AND_INLINE()
inline ::absl::Status FromAbslStatus(const ::absl::Status& s) { return s; }
ABSL_DEPRECATE_AND_INLINE()
inline ::absl::Status ToAbslStatus(const ::absl::Status& s) { return s; }
ABSL_DEPRECATE_AND_INLINE()
inline ::absl::Status OkStatus() { return ::absl::OkStatus(); };
using Status ABSL_DEPRECATE_AND_INLINE() = ::absl::Status;
#endif
using tsl::StatusCallback;
using tsl::StatusGroup;
using tsl::TfCheckOpHelper;
using tsl::TfCheckOpHelperOutOfLine;

namespace errors {
#ifdef SWIG
using tsl::errors::Code;
#else
using Code ABSL_DEPRECATE_AND_INLINE() = ::absl::StatusCode;
#endif
using tsl::errors::GetStackTrace;
using tsl::errors::SetStackTrace;
}  // namespace errors
// NOLINTEND(misc-unused-using-decls)

}  // namespace machina

#endif  // MACHINA_CORE_PLATFORM_STATUS_H_
