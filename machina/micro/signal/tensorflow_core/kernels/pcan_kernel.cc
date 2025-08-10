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

#include "signal/src/pcan_argc_fixed.h"
#include "machina/core/framework/op_kernel.h"
#include "machina/core/framework/types.h"

namespace machina {
namespace signal {

class PcanOp : public machina::OpKernel {
 public:
  explicit PcanOp(machina::OpKernelConstruction* context)
      : machina::OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("snr_shift", &snr_shift_));
  }

  void Compute(machina::OpKernelContext* context) override {
    machina::Tensor* output_tensor = nullptr;
    const uint32_t* input = context->input(0).flat<uint32_t>().data();
    const uint32_t* noise_estimate = context->input(1).flat<uint32_t>().data();
    const int16_t* gain_lut = context->input(2).flat<int16_t>().data();
    int32_t num_channels = context->input(0).NumElements();
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, {num_channels}, &output_tensor));
    uint32_t* output = output_tensor->flat<uint32_t>().data();

    memcpy(output, input, sizeof(uint32_t) * num_channels);
    tflite::tflm_signal::ApplyPcanAutoGainControlFixed(
        gain_lut, snr_shift_, noise_estimate, output, num_channels);
  }

 private:
  int snr_shift_;
};

REGISTER_KERNEL_BUILDER(Name("SignalPCAN").Device(machina::DEVICE_CPU),
                        PcanOp);

}  // namespace signal
}  // namespace machina
