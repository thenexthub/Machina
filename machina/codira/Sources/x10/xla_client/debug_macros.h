/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Sunday, August 10, 2025.
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

#ifndef X10_XLA_CLIENT_DEBUG_MACROS_H_
#define X10_XLA_CLIENT_DEBUG_MACROS_H_

#include "machina/compiler/xla/statusor.h"
#include "machina/compiler/xla/xla_client/tf_logging.h"
#include "machina/core/platform/stacktrace.h"

#define XLA_ERROR() TF_ERROR_STREAM()
#define XLA_CHECK(c) TF_CHECK(c) << "\n" << machina::CurrentStackTrace()
#define XLA_CHECK_OK(c) \
  TF_CHECK_OK(c) << "\n" << machina::CurrentStackTrace()
#define XLA_CHECK_EQ(a, b) \
  TF_CHECK_EQ(a, b) << "\n" << machina::CurrentStackTrace()
#define XLA_CHECK_NE(a, b) \
  TF_CHECK_NE(a, b) << "\n" << machina::CurrentStackTrace()
#define XLA_CHECK_LE(a, b) \
  TF_CHECK_LE(a, b) << "\n" << machina::CurrentStackTrace()
#define XLA_CHECK_GE(a, b) \
  TF_CHECK_GE(a, b) << "\n" << machina::CurrentStackTrace()
#define XLA_CHECK_LT(a, b) \
  TF_CHECK_LT(a, b) << "\n" << machina::CurrentStackTrace()
#define XLA_CHECK_GT(a, b) \
  TF_CHECK_GT(a, b) << "\n" << machina::CurrentStackTrace()

template <typename T>
T ConsumeValue(xla::StatusOr<T>&& status) {
  XLA_CHECK_OK(status.status());
  return status.ConsumeValueOrDie();
}

#endif  // X10_XLA_CLIENT_DEBUG_MACROS_H_
