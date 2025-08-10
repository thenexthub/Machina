/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 26, 2025.
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

#ifndef MACHINA_CORE_KERNELS_IMAGE_RESIZE_BILINEAR_OP_H_
#define MACHINA_CORE_KERNELS_IMAGE_RESIZE_BILINEAR_OP_H_

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "machina/core/framework/numeric_types.h"
#include "machina/core/framework/tensor_types.h"

namespace machina {
namespace functor {

template <typename Device, typename T>
struct ResizeBilinear {
  void operator()(const Device& d, typename TTypes<T, 4>::ConstTensor images,
                  const float height_scale, const float width_scale,
                  const bool half_pixel_centers,
                  typename TTypes<float, 4>::Tensor resized_images);
};

template <typename Device, typename T>
struct ResizeBilinearGrad {
  void operator()(const Device& d,
                  typename TTypes<float, 4>::ConstTensor input_grad,
                  const float height_scale, const float width_scale,
                  const bool half_pixel_centers,
                  typename TTypes<T, 4>::Tensor output_grad);
};

}  // namespace functor
}  // namespace machina

#endif  // MACHINA_CORE_KERNELS_IMAGE_RESIZE_BILINEAR_OP_H_
