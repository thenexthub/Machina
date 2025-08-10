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
#include "machina/core/framework/function.h"
#include "machina/core/framework/op.h"
#include "machina/core/framework/op_kernel.h"
#include "machina/core/framework/tensor.h"
#include "machina/core/framework/types.h"

namespace machina {

namespace {

class GlobalIterId : public OpKernel {
 public:
  explicit GlobalIterId(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) override {
    ctx->set_output(0, Tensor(ctx->frame_iter().iter_id));
  }
};
REGISTER_KERNEL_BUILDER(Name("GlobalIterId").Device(DEVICE_CPU), GlobalIterId);

}  // anonymous namespace
}  // namespace machina
