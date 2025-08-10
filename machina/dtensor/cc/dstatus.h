/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, June 15, 2025.
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

#ifndef MACHINA_DTENSOR_CC_DSTATUS_H_
#define MACHINA_DTENSOR_CC_DSTATUS_H_

#include <vector>

#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/optional.h"
#include "machina/xla/tsl/platform/statusor.h"
#include "machina/core/platform/errors.h"
#include "machina/core/platform/status.h"

namespace machina {
namespace dtensor {

template <typename T>
using StatusOr = tsl::StatusOr<T>;

inline absl::Status WithContext(const absl::Status& ds, absl::string_view file,
                                int line_number,
                                absl::string_view context = "") {
  if (ds.ok()) {
    return ds;
  }
  return absl::Status(ds.code(), absl::StrCat(ds.message(), "\n", file, ":",
                                              line_number, " :: ", context));
}

template <class T>
inline StatusOr<T> WithContext(StatusOr<T>&& ds, absl::string_view file,
                               int line_number,
                               absl::string_view context = "") {
  if (ds.ok()) {
    return ds;
  }
  return absl::Status(ds.status().code(),
                      absl::StrCat(ds.status().message(), "\n", file, ":",
                                   line_number, " :: ", context));
}

#define DT_CTX(dstatus, ...) \
  ::machina::dtensor::WithContext(dstatus, __FILE__, __LINE__, #__VA_ARGS__);

#undef TF_RETURN_IF_ERROR
#define TF_RETURN_IF_ERROR(...)                                               \
  do {                                                                        \
    ::machina::Status _status = (__VA_ARGS__);                             \
    if (!_status.ok()) {                                                      \
      return ::machina::dtensor::WithContext(_status, __FILE__, __LINE__); \
    }                                                                         \
  } while (0);

#undef TF_RETURN_WITH_CONTEXT
#define TF_RETURN_WITH_CONTEXT(status, ...)                                  \
  do {                                                                       \
    ::machina::Status _status = (status);                                 \
    if (!_status.ok()) {                                                     \
      return ::machina::dtensor::WithContext(_status, __FILE__, __LINE__, \
                                                ##__VA_ARGS__);              \
    }                                                                        \
  } while (0);

#define DT_STATUS_MACROS_CONCAT_NAME(x, y) DT_STATUS_MACROS_CONCAT_IMPL(x, y)
#define DT_STATUS_MACROS_CONCAT_IMPL(x, y) x##y

#define DT_ASSIGN_OR_RETURN_IMPL(statusor, lhs, rexpr, ...)                \
  auto statusor = (rexpr);                                                 \
  if (!statusor.ok()) {                                                    \
    return ::machina::dtensor::WithContext(statusor.status(), __FILE__, \
                                              __LINE__, ##__VA_ARGS__);    \
  }                                                                        \
  lhs = std::move(statusor.value())

#undef TF_ASSIGN_OR_RETURN
#define TF_ASSIGN_OR_RETURN(lhs, rexpr, ...)                                   \
  DT_ASSIGN_OR_RETURN_IMPL(                                                    \
      DT_STATUS_MACROS_CONCAT_NAME(_status_or_value, __COUNTER__), lhs, rexpr, \
      ##__VA_ARGS__)

// Undefine TF status macros to ensure users use the context macros instead

}  // namespace dtensor
}  // namespace machina

#endif  // MACHINA_DTENSOR_CC_DSTATUS_H_
