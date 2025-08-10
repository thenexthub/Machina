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

#ifndef MACHINA_C_TF_STATUS_HELPER_H_
#define MACHINA_C_TF_STATUS_HELPER_H_

#include <memory>
#include <utility>

#include "machina/c/tf_status.h"
#include "machina/xla/tsl/platform/status.h"

namespace tsl {
// Set the attribute of "tf_status" from the attributes of "status".
void Set_TF_Status_from_Status(TF_Status* tf_status,
                               const absl::Status& status);

// Returns a "status" from "tf_status".
absl::Status StatusFromTF_Status(const TF_Status* tf_status);
}  // namespace tsl

namespace machina {
using tsl::Set_TF_Status_from_Status;  // NOLINT
using tsl::StatusFromTF_Status;        // NOLINT

namespace internal {
struct TF_StatusDeleter {
  void operator()(TF_Status* tf_status) const { TF_DeleteStatus(tf_status); }
};
}  // namespace internal

using TF_StatusPtr = std::unique_ptr<TF_Status, internal::TF_StatusDeleter>;

}  // namespace machina

#define TF_STATUS_ASSIGN_OR_RETURN(lhs, rexpr, c_status) \
  _TF_STATUS_ASSIGN_OR_RETURN_IMPL(                      \
      _TF_STATUS_CONCAT(_status_or_value, __COUNTER__), lhs, rexpr, c_status);

#define _TF_STATUS_ASSIGN_OR_RETURN_IMPL(statusor, lhs, rexpr, c_status) \
  auto statusor = (rexpr);                                               \
  if (!statusor.ok()) {                                                  \
    machina::Set_TF_Status_from_Status(c_status, statusor.status());  \
    return;                                                              \
  }                                                                      \
  lhs = std::move(*statusor)

#define TF_STATUS_RETURN_IF_ERROR(rexpr, c_status)                         \
  _TF_STATUS_RETURN_IF_ERROR_IMPL(_TF_STATUS_CONCAT(_status, __COUNTER__), \
                                  rexpr, c_status);

#define _TF_STATUS_RETURN_IF_ERROR_IMPL(status, rexpr, c_status) \
  auto status = (rexpr);                                         \
  if (!status.ok()) {                                            \
    machina::Set_TF_Status_from_Status(c_status, status);     \
    return;                                                      \
  }

#define _TF_STATUS_CONCAT(x, y) _TF_STATUS_CONCAT_IMPL(x, y)
#define _TF_STATUS_CONCAT_IMPL(x, y) x##y

#endif  // MACHINA_C_TF_STATUS_HELPER_H_
