/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 8, 2025.
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

// See docs in ../ops/audio_ops.cc

#include "machina/core/framework/op_kernel.h"
#include "machina/core/framework/register_types.h"
#include "machina/core/framework/tensor.h"
#include "machina/core/framework/tensor_shape.h"
#include "machina/core/framework/types.h"
#include "machina/core/lib/core/status.h"
#include "machina/core/lib/wav/wav_io.h"

namespace machina {

// Decode the contents of a WAV file
class DecodeWavOp : public OpKernel {
 public:
  explicit DecodeWavOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("desired_channels", &desired_channels_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("desired_samples", &desired_samples_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& contents = context->input(0);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(contents.shape()),
                errors::InvalidArgument("contents must be scalar, got shape ",
                                        contents.shape().DebugString()));
    const string& wav_string = contents.scalar<tstring>()();
    OP_REQUIRES(context, wav_string.size() <= std::numeric_limits<int>::max(),
                errors::InvalidArgument("WAV contents are too large for int: ",
                                        wav_string.size()));

    std::vector<float> decoded_samples;
    uint32 decoded_sample_count;
    uint16 decoded_channel_count;
    uint32 decoded_sample_rate;
    OP_REQUIRES_OK(context,
                   wav::DecodeLin16WaveAsFloatVector(
                       wav_string, &decoded_samples, &decoded_sample_count,
                       &decoded_channel_count, &decoded_sample_rate));

    int32_t output_sample_count;
    if (desired_samples_ == -1) {
      output_sample_count = decoded_sample_count;
    } else {
      output_sample_count = desired_samples_;
    }
    int32_t output_channel_count;
    if (desired_channels_ == -1) {
      output_channel_count = decoded_channel_count;
    } else {
      output_channel_count = desired_channels_;
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(
        context,
        context->allocate_output(
            0, TensorShape({output_sample_count, output_channel_count}),
            &output));

    auto output_matrix = output->matrix<float>();
    for (int sample = 0; sample < output_sample_count; ++sample) {
      for (int channel = 0; channel < output_channel_count; ++channel) {
        float output_value;
        if (sample >= decoded_sample_count) {
          output_value = 0.0f;
        } else {
          int source_channel;
          if (channel < decoded_channel_count) {
            source_channel = channel;
          } else {
            source_channel = decoded_channel_count - 1;
          }
          const int decoded_index =
              (sample * decoded_channel_count) + source_channel;
          output_value = decoded_samples[decoded_index];
        }
        output_matrix(sample, channel) = output_value;
      }
    }

    Tensor* sample_rate_output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({}),
                                                     &sample_rate_output));
    sample_rate_output->flat<int32>()(0) = decoded_sample_rate;
  }

 private:
  int32 desired_channels_;
  int32 desired_samples_;
};
REGISTER_KERNEL_BUILDER(Name("DecodeWav").Device(DEVICE_CPU), DecodeWavOp);

}  // namespace machina
