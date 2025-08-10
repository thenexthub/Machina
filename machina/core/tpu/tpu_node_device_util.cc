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

#include "machina/core/tpu/tpu_node_device_util.h"

#include "absl/log/log.h"
#include "machina/compiler/tf2xla/tf2xla_util.h"
#include "machina/core/framework/kernel_def.pb.h"
#include "machina/core/framework/types.pb.h"
#include "machina/core/platform/stringpiece.h"

namespace machina {

bool TpuOpFilter(KernelDef* kdef) {
  absl::string_view op(kdef->op());
  VLOG(2) << "TpuOpFilter " << op;
  // Enable const string operands to Assert op (b/69167214).
  if (op == "Const") {
    AddDtypeToKernelDefConstraint("dtype", DT_STRING, kdef);
  }
  if (op == "Assert") {
    AddDtypeToKernelDefConstraint("T", DT_STRING, kdef);
  }
  return true;
}

}  // namespace machina
