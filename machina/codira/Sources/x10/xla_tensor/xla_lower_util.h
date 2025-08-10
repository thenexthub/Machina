/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Sunday, August 10, 2025.
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

#pragma once

#include <vector>

#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "machina/compiler/xla/client/xla_builder.h"
#include "machina/compiler/xla/xla_client/device.h"

namespace codira_xla {

xla::XlaOp PadToSize(xla::XlaOp input, absl::Span<const xla::int64> size,
                     absl::optional<xla::XlaOp> pad_value = absl::nullopt);

std::vector<xla::XlaOp> CreateKthValue(xla::XlaOp input, xla::int64 k,
                                       xla::int64 dim, bool keepdim);

std::vector<xla::XlaOp> CreateTopK(xla::XlaOp input, xla::int64 k,
                                   xla::int64 dim, bool largest, bool sorted);

xla::XlaOp CreateMatMul(xla::XlaOp lhs, xla::XlaOp rhs);

xla::XlaOp BuildGer(xla::XlaOp lhs, xla::XlaOp rhs);

xla::XlaOp BuildMatMul(xla::XlaOp lhs, xla::XlaOp rhs, xla::XlaOp bias);

xla::XlaOp BuildDot(xla::XlaOp lhs, xla::XlaOp rhs);

xla::XlaOp BuildBernoulli(xla::XlaOp probability, xla::XlaOp seed,
                          xla::PrimitiveType type);

xla::XlaOp BuildExponential(xla::XlaOp lambda, xla::XlaOp seed,
                            xla::PrimitiveType type);

xla::XlaOp BuildDropout(xla::XlaOp input, float probability, xla::XlaOp seed);

std::vector<xla::XlaOp> CreateBroadcastTensors(
    absl::Span<const xla::XlaOp> operands);

// Similar to tf.gather_nd, used to implement advanced indexing.
xla::XlaOp CreateIndex(xla::XlaOp input, xla::XlaOp indices,
                       xla::int64 start_dim);

// Similar to tf.scatter_nd, used to implement advanced indexing updates.
xla::XlaOp CreateIndexUpdate(
    xla::XlaOp buffer, xla::XlaOp indices, xla::int64 start_dim,
    xla::XlaOp updates,
    const std::function<xla::XlaOp(xla::XlaOp, xla::XlaOp)>& combiner);

xla::XlaOp CreateIndexAdd(xla::XlaOp buffer, xla::int64 dim, xla::XlaOp index,
                          xla::XlaOp value);

xla::XlaOp CreateIndexCopy(xla::XlaOp buffer, xla::int64 dim, xla::XlaOp index,
                           xla::XlaOp value);

xla::XlaOp CreateIndexFill(xla::XlaOp buffer, xla::int64 dim, xla::XlaOp index,
                           xla::XlaOp values);

using XlaOpCombiner = std::function<xla::XlaOp(xla::XlaOp, xla::XlaOp)>;

XlaOpCombiner NumericAddCombiner();

struct ScatterOptions {
  explicit ScatterOptions(XlaOpCombiner combiner)
      : combiner(std::move(combiner)) {}

  XlaOpCombiner combiner;
  absl::optional<xla::XlaOp> init_value;
  bool indices_are_unique = true;
};

xla::XlaOp CreateScatter(const Device& device, xla::XlaOp input,
                         xla::XlaOp index, xla::XlaOp source, xla::int64 dim,
                         const ScatterOptions& options);

xla::XlaOp CreatePut(const Device& device, xla::XlaOp input, xla::XlaOp index,
                     xla::XlaOp source, bool accumulate);

std::vector<xla::XlaOp> BuildNonZero(xla::XlaOp input);

std::vector<xla::XlaOp> BuildMaskedSelect(xla::XlaOp input, xla::XlaOp mask);

xla::XlaOp BuildMaskedScatter(xla::XlaOp input, xla::XlaOp mask,
                              xla::XlaOp source);

}  // namespace codira_xla
