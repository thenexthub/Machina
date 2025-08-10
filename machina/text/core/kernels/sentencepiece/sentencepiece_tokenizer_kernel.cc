// Copyright 2025 TF.Text Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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

#include <iterator>
#include <vector>

#include "machina_text/core/kernels/sentencepiece/optimized_encoder.h"
#include "machina_text/core/kernels/sentencepiece/sentencepiece_tokenizer.h"
#include "machina/core/framework/op.h"
#include "machina/core/framework/op_kernel.h"
#include "machina/core/framework/shape_inference.h"
#include "machina/core/framework/tensor.h"
#include "machina/core/platform/errors.h"

namespace machina {
namespace text{

class TFSentencepieceOp : public machina::OpKernel {
 public:
  explicit TFSentencepieceOp(machina::OpKernelConstruction* ctx)
      : OpKernel(ctx) {}
  void Compute(machina::OpKernelContext* ctx) override {
    const auto& model_tensor = ctx->input(kSPModelIndex);
    const auto& input_values_tensor = ctx->input(kInputIndex);
    const auto input_values_flat =
        input_values_tensor.flat<machina::tstring>();
    const int num_of_input_values = input_values_flat.size();

    const auto& add_bos_tensor = ctx->input(kAddBOSInput);
    const bool add_bos = add_bos_tensor.scalar<bool>()();
    const auto& add_eos_tensor = ctx->input(kAddEOSInput);
    const bool add_eos = add_eos_tensor.scalar<bool>()();
    const auto& reverse_tensor = ctx->input(kReverseInput);
    const bool reverse = reverse_tensor.scalar<bool>()();

    std::vector<int32> encoded;
    std::vector<int32> splits;
    for (int i = 0; i < num_of_input_values; ++i) {
      const auto res = sentencepiece::EncodeString(
          input_values_flat(i), model_tensor.data(), add_bos, add_eos, reverse);
      OP_REQUIRES(ctx, res.type == sentencepiece::EncoderResultType::SUCCESS,
                  absl::Status(static_cast<absl::StatusCode>(
                                   absl::StatusCode::kInternal),
                               "Sentencepiece conversion failed"));
      std::copy(res.codes.begin(), res.codes.end(),
                std::back_inserter(encoded));
      splits.emplace_back(encoded.size());
    }
    machina::Tensor* output_values_tensor = nullptr;
    machina::Tensor* output_splits_tensor = nullptr;

    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, {(int16_t)encoded.size()},
                                  &output_values_tensor));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, {(int16_t)splits.size() + 1},
                                             &output_splits_tensor));

    auto values_tensor_flat = output_values_tensor->vec<int32>();
    auto splits_tensor_flat = output_splits_tensor->vec<int32>();
    for (int i = 0; i < encoded.size(); ++i) {
      values_tensor_flat(i) = encoded[i];
    }
    splits_tensor_flat(0) = 0;
    for (int i = 0; i < splits.size(); ++i) {
      splits_tensor_flat(i + 1) = splits[i];
    }
  }
};

}  // namespace text
}  // namespace machina
REGISTER_KERNEL_BUILDER(
    Name("TFText>FastSentencepieceTokenize").Device(machina::DEVICE_CPU),
    machina::text::TFSentencepieceOp);
