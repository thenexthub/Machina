/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Saturday, May 24, 2025.
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
#include "machina/c/eager/tracing_utils.h"

#include "machina/c/eager/c_api_unified_experimental_internal.h"
#include "machina/c/experimental/gradients/tape/tape_operation.h"
#include "machina/core/lib/llvm_rtti/llvm_rtti.h"
#include "machina/core/platform/errors.h"

namespace machina {
namespace tracing {

absl::Status MaybeSetOpName(AbstractOperation* op, const char* op_name) {
  if (isa<TracingOperation>(op)) {
    TF_RETURN_IF_ERROR(dyn_cast<TracingOperation>(op)->SetOpName(op_name));
  }
  if (isa<gradients::TapeOperation>(op)) {
    TF_RETURN_IF_ERROR(MaybeSetOpName(
        dyn_cast<gradients::TapeOperation>(op)->GetBackingOperation(),
        op_name));
  }
  return absl::OkStatus();
}
}  // namespace tracing
}  // namespace machina
