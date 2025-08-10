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

#include "machina/core/framework/bounds_check.h"
#include "machina/core/framework/op_kernel.h"
#include "machina/core/framework/register_types.h"
#include "machina/core/framework/tensor.h"
#include "machina/core/framework/tensor_shape.h"
#include "machina/core/framework/types.h"
#include "machina/core/lib/core/status.h"
#include "machina/core/lib/wav/wav_io.h"

namespace machina {

// Encode a tensor as audio samples into the contents of a WAV format file.
class EncodeWavOp : public OpKernel {
 public:
  explicit EncodeWavOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& audio = context->input(0);
    OP_REQUIRES(context, audio.dims() == 2,
                errors::InvalidArgument("audio must be 2-dimensional",
                                        audio.shape().DebugString()));
    const Tensor& sample_rate_tensor = context->input(1);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(sample_rate_tensor.shape()),
                errors::InvalidArgument(
                    "Input sample_rate should be a scalar tensor, got ",
                    sample_rate_tensor.shape().DebugString(), " instead."));
    const int32_t sample_rate = sample_rate_tensor.scalar<int32>()();
    OP_REQUIRES(
        context,
        FastBoundsCheck(audio.NumElements(), std::numeric_limits<int32>::max()),
        errors::InvalidArgument(
            "Cannot encode audio with >= max int32 elements"));

    const int32_t channel_count = static_cast<int32>(audio.dim_size(1));
    const int32_t sample_count = static_cast<int32>(audio.dim_size(0));

    // Encode audio to wav string.
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({}), &output));
    OP_REQUIRES_OK(context,
                   wav::EncodeAudioAsS16LEWav(
                       audio.flat<float>().data(), sample_rate, channel_count,
                       sample_count, &output->scalar<tstring>()()));
  }
};
REGISTER_KERNEL_BUILDER(Name("EncodeWav").Device(DEVICE_CPU), EncodeWavOp);

}  // namespace machina
