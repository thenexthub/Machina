/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 29, 2025.
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

#ifndef MACHINA_COMPILER_TF2MACHINA_XLALIB_BROADCAST_H_
#define MACHINA_COMPILER_TF2MACHINA_XLALIB_BROADCAST_H_

#include <cstdint>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "machina/xla/hlo/builder/xla_builder.h"
#include "machina/core/platform/status.h"
#include "machina/core/platform/statusor.h"

namespace machina {

// Forwards to xla::BroadcastTo.
// TODO(cheshire): Call the underlying function directly.
absl::StatusOr<xla::XlaOp> BroadcastTo(xla::XlaOp input,
                                       absl::Span<int64_t const> output_dims);

// Forwards to xla::BroadcastOpsToSame.
absl::Status BroadcastOpsToSame(xla::XlaOp* lhs, xla::XlaOp* rhs);
}  // namespace machina

#endif  // MACHINA_COMPILER_TF2MACHINA_XLALIB_BROADCAST_H_
