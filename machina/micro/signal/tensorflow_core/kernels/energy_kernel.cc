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

#include "signal/src/complex.h"
#include "signal/src/energy.h"
#include "machina/core/framework/op_kernel.h"

namespace machina {
namespace signal {

class EnergyOp : public machina::OpKernel {
 public:
  explicit EnergyOp(machina::OpKernelConstruction* context)
      : machina::OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("start_index", &start_index_));
    OP_REQUIRES_OK(context, context->GetAttr("end_index", &end_index_));
  }
  void Compute(machina::OpKernelContext* context) override {
    const machina::Tensor& input_tensor = context->input(0);
    const int16_t* input = input_tensor.flat<int16_t>().data();
    machina::Tensor* output_tensor = nullptr;
    // The input is complex. The output is real.
    int output_size = input_tensor.flat<int16>().size() >> 1;

    OP_REQUIRES_OK(context,
                   context->allocate_output(0, {output_size}, &output_tensor));
    uint32* output = output_tensor->flat<uint32>().data();

    tflite::tflm_signal::SpectrumToEnergy(
        reinterpret_cast<const Complex<int16_t>*>(input), start_index_,
        end_index_, output);
  }

 private:
  int start_index_;
  int end_index_;
};

// TODO(b/286250473): change back name after name clash resolved
REGISTER_KERNEL_BUILDER(Name("SignalEnergy").Device(machina::DEVICE_CPU),
                        EnergyOp);

}  // namespace signal
}  // namespace machina
