/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, June 15, 2025.
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

#include "signal/src/window.h"
#include "machina/core/framework/op_kernel.h"

namespace machina {
namespace signal {

class WindowOp : public machina::OpKernel {
 public:
  explicit WindowOp(machina::OpKernelConstruction* context)
      : machina::OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("shift", &shift_));
  }

  void Compute(machina::OpKernelContext* context) override {
    const machina::Tensor& input_tensor = context->input(0);
    const int16_t* input = input_tensor.flat<int16_t>().data();
    const machina::Tensor& weight_tensor = context->input(1);
    const int16_t* weights = weight_tensor.flat<int16_t>().data();
    int weight_size = weight_tensor.flat<int16_t>().size();
    int outer_dims =
        input_tensor.flat_inner_dims<int16_t, 2>().dimensions().at(0);

    machina::Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    int16_t* output = output_tensor->flat<int16_t>().data();
    for (int i = 0; i < outer_dims; i++) {
      tflm_signal::ApplyWindow(&input[i * weight_size], weights, weight_size,
                               shift_, &output[i * weight_size]);
    }
  }

 private:
  int shift_;
};

// TODO(b/286250473): change back name to "Window" after name clash resolved
REGISTER_KERNEL_BUILDER(Name("SignalWindow").Device(machina::DEVICE_CPU),
                        WindowOp);

}  // namespace signal
}  // namespace machina