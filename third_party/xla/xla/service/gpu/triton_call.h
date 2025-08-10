/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Friday, August 8, 2025.
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

#ifndef MACHINA_XLASERVICE_GPU_TRITON_CALL_H_
#define MACHINA_XLASERVICE_GPU_TRITON_CALL_H_

#include <cstdint>
#include <string>

#include "absl/strings/string_view.h"
#include "mlir/IR/MLIRContext.h"

namespace xla::gpu {

struct TritonCall {
  std::string name;
  std::string ir;
  int64_t num_stages;
  int64_t num_warps;
  int32_t grid_x;
  int32_t grid_y;
  int32_t grid_z;

  // Parse the metadata of a __gpu$xla.gpu.triton call.
  static TritonCall Parse(absl::string_view backend_config,
                          mlir::MLIRContext* mlir_context);
};

}  // namespace xla::gpu

#endif  // MACHINA_XLASERVICE_GPU_TRITON_CALL_H_
