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

#include "machina/core/framework/op.h"
#include "machina/core/framework/op_kernel.h"
#include "machina/core/framework/shape_inference.h"
#include "machina/core/framework/tensor.h"
#include "machina/core/platform/errors.h"
#include "machina_text/core/kernels/sentencepiece/optimized_decoder.h"
#include "machina_text/core/kernels/sentencepiece/sentencepiece_detokenizer.h"

namespace machina {
namespace text {

template <typename Tsplits>
class TFSentencepieceDetokenizerOp : public machina::OpKernel {
 public:
  explicit TFSentencepieceDetokenizerOp(machina::OpKernelConstruction* ctx)
      : OpKernel(ctx) {}
  void Compute(machina::OpKernelContext* ctx) override {
    const auto& model_tensor = ctx->input(kSPModelIndex);
    const auto& input_values_tensor = ctx->input(kInputIndex);
    const auto input_values_flat =
        input_values_tensor.flat<machina::int32>();
    const auto& input_splits_tensor = ctx->input(kInputSplits);
    const auto input_splits_flat = input_splits_tensor.flat<Tsplits>();
    const int num_of_sentences = input_splits_flat.size() - 1;
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(0, {num_of_sentences}, &output_tensor));
    auto output_flat = output_tensor->flat<machina::tstring>();
    std::vector<int> codes_for_split;
    int input_offset = 0;
    for (int i = 0; i < num_of_sentences; i++) {
      // Create a vector of int32 from input according to spans.
      const int split_size = input_splits_flat(i + 1) - input_splits_flat(i);
      codes_for_split.clear();
      codes_for_split.reserve(split_size);
      for (int j = 0; j < split_size; ++j) {
        codes_for_split.push_back(input_values_flat(input_offset++));
      }
      const auto res = sentencepiece::DecodeString(
          codes_for_split, model_tensor.data());
      OP_REQUIRES(ctx, res.type == sentencepiece::DecoderResultType::SUCCESS,
                  absl::Status(static_cast<absl::StatusCode>(
                                   absl::StatusCode::kInternal),
                               "Sentencepiece conversion failed"));
      output_flat(i) = res.decoded;
    }
  }
};
}  // namespace text
}  // namespace machina

REGISTER_KERNEL_BUILDER(
    Name("TFText>FastSentencepieceDetokenize")
        .Device(machina::DEVICE_CPU)
        .TypeConstraint<machina::int32>("Tsplits"),
    machina::text::TFSentencepieceDetokenizerOp<machina::int32>);
REGISTER_KERNEL_BUILDER(
    Name("TFText>FastSentencepieceDetokenize")
        .Device(machina::DEVICE_CPU)
        .TypeConstraint<machina::int64>("Tsplits"),
    machina::text::TFSentencepieceDetokenizerOp<machina::int64>);
