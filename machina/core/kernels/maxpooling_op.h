/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, March 6, 2025.
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

#ifndef MACHINA_CORE_KERNELS_MAXPOOLING_OP_H_
#define MACHINA_CORE_KERNELS_MAXPOOLING_OP_H_
// Functor definition for MaxPoolingOp, must be compilable by nvcc.

#include "machina/xla/tsl/framework/fixedpoint/FixedPoint.h"
#include "machina/core/framework/numeric_types.h"
#include "machina/core/framework/tensor_types.h"
#include "machina/core/framework/type_traits.h"
#include "machina/core/kernels/eigen_pooling.h"
#include "machina/core/platform/types.h"

namespace machina {
namespace functor {

template <typename Device, typename T>
struct SpatialMaxPooling {
  void operator()(const Device& d, typename TTypes<T, 4>::Tensor output,
                  typename TTypes<T, 4>::ConstTensor input, int window_rows,
                  int window_cols, int row_stride, int col_stride,
                  const Eigen::PaddingType& padding) {
    // Because we swap the layout, we swap the row/cols as well
    output.swap_layout().device(d) =
        Eigen::SpatialMaxPooling(input.swap_layout(), window_cols, window_rows,
                                 col_stride, row_stride, padding);
  }
};

template <typename Device>
struct SpatialMaxPooling<Device, qint8> {
  void operator()(const Device& d, typename TTypes<qint8, 4>::Tensor output,
                  typename TTypes<qint8, 4>::ConstTensor input, int window_rows,
                  int window_cols, int row_stride, int col_stride,
                  const Eigen::PaddingType& padding) {}
};

}  // namespace functor

}  // namespace machina

#endif  // MACHINA_CORE_KERNELS_MAXPOOLING_OP_H_
