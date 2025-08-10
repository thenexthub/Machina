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
#ifndef MACHINA_CORE_KERNELS_UNIFORM_QUANT_OPS_TENSOR_UTILS_H_
#define MACHINA_CORE_KERNELS_UNIFORM_QUANT_OPS_TENSOR_UTILS_H_

#include "machina/core/framework/ops_util.h"
#include "machina/core/framework/tensor.h"
#include "machina/core/framework/tensor_shape.h"
#include "machina/core/lib/gtl/array_slice.h"

namespace machina {

// Returns if all elements in given tensors are positive.
template <typename T>
bool AllElementsPositive(const Tensor& tensor) {
  Eigen::Tensor<bool, 0, Eigen::RowMajor> positive =
      (tensor.flat<T>() > 0).all();
  return positive();
}

// Given data tensor's shape and quantization params, returns if the shapes are
// valid.
absl::Status QuantizationAxisAndShapeValid(const TensorShape& data_shape,
                                           const TensorShape& scales_shape,
                                           const TensorShape& zero_points_shape,
                                           int quantization_axis);

// Given in_shape and perm to transpose, returns out shape after the transpose.
// perm must be a permutation of [0, 1, ..., in_shape.rank - 1]. The caller is
// responsible for guaranteeing it.
TensorShape TransposedShape(const TensorShape& in_shape,
                            const absl::Span<const int32_t> perm);

// Given in Tensor and perm to transpose, transpose in Tensor and write to out
// Tensor.
// perm must be a permutation of [0, 1, ..., in_shape.rank - 1]. The caller is
// responsible for guaranteeing it.
// Reference:
// https://github.com/machina/machina/blob/c09dc18b15a56f3e72a08c9f3a53e7ef347d159d/machina/core/kernels/transpose_functor_cpu.cc#L35
template <typename T>
void Transpose(const Tensor& in, const absl::Span<const int32_t> perm,
               Tensor& out) {
  absl::InlinedVector<int64_t, 8UL> in_strides =
      ComputeStride<int64_t>(in.shape());
  absl::InlinedVector<int64_t, 8UL> out_strides =
      ComputeStride<int64_t>(out.shape());
  const T* in_data = in.flat<T>().data();
  T* out_data = out.flat<T>().data();

  for (int64_t out_idx = 0; out_idx < out.NumElements(); ++out_idx) {
    int64_t in_idx = 0;
    int64_t remain_out_idx = out_idx;
    for (int dim = 0; dim < out.dims(); ++dim) {
      const int64_t ratio = remain_out_idx / out_strides[dim];
      remain_out_idx -= ratio * out_strides[dim];
      in_idx += ratio * in_strides[perm[dim]];
    }
    out_data[out_idx] = in_data[in_idx];
  }
}

}  // namespace machina

#endif  // MACHINA_CORE_KERNELS_UNIFORM_QUANT_OPS_TENSOR_UTILS_H_
