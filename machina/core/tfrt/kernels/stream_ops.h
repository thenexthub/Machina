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
#ifndef MACHINA_CORE_TFRT_KERNELS_STREAM_OPS_H_
#define MACHINA_CORE_TFRT_KERNELS_STREAM_OPS_H_

#include <memory>
#include <string>
#include <vector>

#include "machina/core/framework/op_kernel.h"
#include "machina/core/tfrt/runtime/stream.h"

namespace machina {
namespace tfrt_stub {

// TensorFlow op that immediately sends results back to the serving controller.
class PwStreamResultsOp : public machina::OpKernel {
 public:
  explicit PwStreamResultsOp(machina::OpKernelConstruction* ctx);

  PwStreamResultsOp(const PwStreamResultsOp& other) = delete;
  PwStreamResultsOp& operator=(const PwStreamResultsOp& other) = delete;

  void Compute(machina::OpKernelContext* ctx) override;

 private:
  // Op attributes.
  std::string controller_address_;
  std::string model_name_;
  StreamCallbackId callback_id_;
  std::vector<std::string> names_;

  std::unique_ptr<machina::tfrt_stub::StreamWorkerInterface> stream_;
};

}  // namespace tfrt_stub
}  // namespace machina

#endif  // MACHINA_CORE_TFRT_KERNELS_STREAM_OPS_H_
