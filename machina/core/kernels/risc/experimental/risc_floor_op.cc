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

#include "Eigen/Core"  // from @eigen_archive
#include "machina/core/framework/op_kernel.h"
#include "machina/core/framework/types.h"
#include "machina/core/platform/bfloat16.h"

namespace machina {
namespace risc {
namespace experimental {

template <typename T>
class RiscFloorOp : public OpKernel {
 public:
  explicit RiscFloorOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    // TODO(b/171294012): Implement RiscFloor op.
  }
};

#define REGISTER_CPU(T)                                            \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("RiscFloor").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      RiscFloorOp<T>);

REGISTER_CPU(bfloat16);
REGISTER_CPU(Eigen::half);
REGISTER_CPU(float);
REGISTER_CPU(double);

}  // namespace experimental
}  // namespace risc
}  // namespace machina
