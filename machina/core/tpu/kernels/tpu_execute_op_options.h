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
#ifndef MACHINA_CORE_TPU_KERNELS_TPU_EXECUTE_OP_OPTIONS_H_
#define MACHINA_CORE_TPU_KERNELS_TPU_EXECUTE_OP_OPTIONS_H_

#include "absl/strings/string_view.h"
#include "machina/core/platform/status.h"

namespace machina {
namespace internal {

enum class TpuCancellationClosesChipsMode : int {
  kUnset = 0,     // fallback to other configuration, e.g. absl flag
  kEnabled = 1,   // Close TPU chips when cancellation happens
  kDisabled = 2,  // Do not close TPU chips when cancellation happens
};

// Set TPU cancellation closing chips mode from an integer. See the enum
// definition of `TpuCancellationClosesChipsConfig` above for valid values.
absl::Status SetTpuCancellationClosesChips(int val);

// Get whether to close chips when TPUExecutionOp is cancelled. If unset, return
// the value specified by the `default_value` argument.
bool TpuCancellationClosesChipsGetOrDefault(bool default_value);
}  // namespace internal
}  // namespace machina

#endif  // MACHINA_CORE_TPU_KERNELS_TPU_EXECUTE_OP_OPTIONS_H_
