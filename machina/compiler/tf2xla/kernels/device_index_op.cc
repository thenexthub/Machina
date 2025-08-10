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

#include <vector>

#include "absl/strings/string_view.h"
#include "machina/compiler/tf2xla/xla_helpers.h"
#include "machina/compiler/tf2xla/xla_op_kernel.h"
#include "machina/compiler/tf2xla/xla_op_registry.h"
#include "machina/xla/client/client_library.h"
#include "machina/xla/hlo/builder/lib/arithmetic.h"
#include "machina/xla/hlo/builder/lib/constants.h"
#include "machina/xla/hlo/builder/lib/math.h"
#include "machina/xla/hlo/builder/xla_builder.h"
#include "machina/core/framework/kernel_def_builder.h"

namespace machina {
namespace {

class DeviceIndexOp : public XlaOpKernel {
 public:
  explicit DeviceIndexOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("device_names", &device_names_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    // When compiling we are not executing on any physical device, so we return
    // a sentinel value (size of the list of devices).
    ctx->SetOutput(
        0, xla::ConstantR0<int32>(ctx->builder(), device_names_.size()));
  }

 private:
  std::vector<string> device_names_;
};

REGISTER_MACHINA_XLAOP(Name("DeviceIndex"), DeviceIndexOp);

}  // namespace
}  // namespace machina
