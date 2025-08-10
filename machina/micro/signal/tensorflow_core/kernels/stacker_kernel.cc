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

#include "signal/src/circular_buffer.h"
#include "machina/core/framework/op_kernel.h"

namespace machina {
namespace signal {

class StackerOp : public machina::OpKernel {
 public:
  explicit StackerOp(machina::OpKernelConstruction* context)
      : machina::OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("num_channels", &num_channels_));
    OP_REQUIRES_OK(context, context->GetAttr("stacker_left_context",
                                             &stacker_left_context_));
    OP_REQUIRES_OK(context, context->GetAttr("stacker_right_context",
                                             &stacker_right_context_));
    OP_REQUIRES_OK(context, context->GetAttr("stacker_step", &stacker_step_));
    buffer_size_ =
        num_channels_ * (stacker_left_context_ + stacker_right_context_ + 1);
    step_size_ = num_channels_ * stacker_step_;
    stacker_has_first_frame_ = false;

    size_t state_size =
        tflite::tflm_signal::CircularBufferGetNeededMemory(buffer_size_);
    OP_REQUIRES_OK(context,
                   context->allocate_temp(
                       DT_INT8, TensorShape({static_cast<int32_t>(state_size)}),
                       &state_tensor_));
    state_ = state_tensor_.flat<int8_t>().data();
    circular_buffer = tflite::tflm_signal::CircularBufferInit(
        buffer_size_, state_, state_size);
  }

  void Compute(machina::OpKernelContext* context) override {
    const machina::Tensor& input_tensor = context->input(0);
    const int16_t* input = input_tensor.flat<int16_t>().data();

    tflite::tflm_signal::CircularBufferWrite(circular_buffer, input,
                                             num_channels_);

    // The first frame is replicated an extra left_context times to pad.
    if (stacker_has_first_frame_ == false) {
      tflite::tflm_signal::CircularBufferExtend(circular_buffer, num_channels_,
                                                stacker_left_context_);
      stacker_has_first_frame_ = true;
    }

    machina::Tensor* output_tensor = nullptr;
    machina::Tensor* output_valid_tensor = nullptr;

    OP_REQUIRES_OK(
        context, context->allocate_output(
                     0, {static_cast<int32_t>(buffer_size_)}, &output_tensor));
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, {}, &output_valid_tensor));

    if (tflite::tflm_signal::CircularBufferAvailable(circular_buffer) >=
        buffer_size_) {
      tflite::tflm_signal::CircularBufferGet(
          circular_buffer, buffer_size_, output_tensor->flat<int16_t>().data());
      tflite::tflm_signal::CircularBufferDiscard(circular_buffer, step_size_);
      *output_valid_tensor->flat<bool>().data() = true;
    } else {
      *output_valid_tensor->flat<bool>().data() = false;
    }
  }

 private:
  int num_channels_;
  int stacker_left_context_;
  int stacker_right_context_;
  int stacker_step_;
  size_t buffer_size_;
  size_t step_size_;
  bool stacker_has_first_frame_;

  int8_t* state_;
  Tensor state_tensor_;
  tflite::tflm_signal::CircularBuffer* circular_buffer;
};

REGISTER_KERNEL_BUILDER(Name("SignalStacker").Device(machina::DEVICE_CPU),
                        StackerOp);

}  // namespace signal
}  // namespace machina
