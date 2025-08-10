/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, July 12, 2025.
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

#ifndef MACHINA_CORE_TFRT_KERNELS_IFRT_PROGRAM_OPS_H_
#define MACHINA_CORE_TFRT_KERNELS_IFRT_PROGRAM_OPS_H_

#include <stdint.h>

#include <string>
#include <vector>

#include "absl/base/call_once.h"
#include "machina/core/framework/op_kernel.h"
#include "machina/core/tfrt/ifrt/ifrt_serving_executable.h"

namespace machina {
namespace tfrt_stub {

// TensorFlow op that calls a Ifrt program registered in `ProgramRegistry`.
class IfrtCallOp : public machina::OpKernel {
 public:
  explicit IfrtCallOp(machina::OpKernelConstruction* ctx);

  IfrtCallOp(const IfrtCallOp& other) = delete;
  IfrtCallOp& operator=(const IfrtCallOp& other) = delete;

  void Compute(machina::OpKernelContext* ctx) override;

 private:
  // Op attributes.
  int64_t program_id_;

  std::vector<std::string> variable_names_;
  std::vector<int> variable_arg_indices_;

  // Ifrt program to be called. Cached after the first call.
  absl::once_flag init_once_;
  machina::ifrt_serving::IfrtServingExecutable* executable_;  // Not owned.
};

}  // namespace tfrt_stub
}  // namespace machina
#endif  // MACHINA_CORE_TFRT_KERNELS_IFRT_PROGRAM_OPS_H_
