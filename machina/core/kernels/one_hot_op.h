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

// See docs in ../ops/array_ops.cc

#ifndef MACHINA_CORE_KERNELS_ONE_HOT_OP_H_
#define MACHINA_CORE_KERNELS_ONE_HOT_OP_H_
// Generator definition for OneHotOp, must be compilable by nvcc.

#define EIGEN_USE_THREADS

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "machina/core/framework/bounds_check.h"
#include "machina/core/framework/tensor_types.h"
#include "machina/core/platform/macros.h"
#include "machina/core/platform/types.h"

namespace machina {

typedef Eigen::ThreadPoolDevice CPUDevice;

namespace generator {

template <typename T, typename TI>
class OneGenerator {
 public:
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
  OneGenerator(const typename TTypes<TI>::ConstMatrix& indices,
               const typename TTypes<T>::ConstScalar& on_value,
               const typename TTypes<T>::ConstScalar& off_value)
      : indices_(indices), on_value_(on_value), off_value_(off_value) {}

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE T
  operator()(const Eigen::array<Eigen::DenseIndex, 3>& pre_depth_suff) const {
    return (indices_(pre_depth_suff[0], pre_depth_suff[2]) == pre_depth_suff[1])
               ? on_value_()
               : off_value_();
  }

 private:
  const typename TTypes<TI>::ConstMatrix indices_;
  const typename TTypes<T>::ConstScalar on_value_;
  const typename TTypes<T>::ConstScalar off_value_;
};

}  // namespace generator

namespace functor {

template <typename Device, typename T, typename TI>
struct OneHot {
  EIGEN_ALWAYS_INLINE static void Compute(
      const Device& d, const typename TTypes<TI>::ConstMatrix& indices,
      const typename TTypes<T>::ConstScalar& on_value,
      const typename TTypes<T>::ConstScalar& off_value,
      typename TTypes<T, 3>::Tensor* output) {
    generator::OneGenerator<T, TI> generator(indices, on_value, off_value);
    output->device(d) = output->generate(generator);
  }
};

template <typename T, typename TI>
struct OneHot<CPUDevice, T, TI> {
  EIGEN_ALWAYS_INLINE static void Compute(
      const CPUDevice& d, const typename TTypes<TI>::ConstMatrix& indices,
      const typename TTypes<T>::ConstScalar& on_value,
      const typename TTypes<T>::ConstScalar& off_value,
      typename TTypes<T, 3>::Tensor* output) {
    // Pre-fill output with `off_value`.
    output->device(d) = output->constant(off_value());

    // Iterate through indices and update on_value elements in the output.
    Eigen::Index prefix_size = output->dimensions()[0];
    Eigen::Index depth_size = output->dimensions()[1];
    Eigen::Index suffix_size = output->dimensions()[2];

    // Cost of setting one `on_value` coefficient.
    double bytes_loaded = sizeof(T);
    double bytes_stored = sizeof(T);
    double cycles = 0.0;
    const Eigen::TensorOpCost cost(bytes_loaded, bytes_stored, cycles);

    if (suffix_size == 1) {
      const auto func = [&](Eigen::Index start, Eigen::Index end) -> void {
        for (Eigen::Index i = start; i < end; ++i) {
          const TI depth = internal::SubtleMustCopy(indices(i, 0));
          if (FastBoundsCheck(depth, depth_size)) {
            (*output)(i, depth, 0) = on_value();
          }
        }
      };
      d.parallelFor(prefix_size, cost, func);
    } else {
      const auto func = [&](Eigen::Index start, Eigen::Index end) -> void {
        for (Eigen::Index i = start; i < end; ++i) {
          const Eigen::Index d0 = i / suffix_size;
          const Eigen::Index d1 = i - (d0 * suffix_size);
          const TI depth = internal::SubtleMustCopy(indices(d0, d1));
          if (FastBoundsCheck(depth, depth_size)) {
            (*output)(d0, depth, d1) = on_value();
          }
        }
      };
      d.parallelFor(prefix_size * suffix_size, cost * suffix_size, func);
    }
  }
};

}  // namespace functor

}  // namespace machina

#endif  // MACHINA_CORE_KERNELS_ONE_HOT_OP_H_
