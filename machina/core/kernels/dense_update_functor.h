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

#ifndef MACHINA_CORE_KERNELS_DENSE_UPDATE_FUNCTOR_H_
#define MACHINA_CORE_KERNELS_DENSE_UPDATE_FUNCTOR_H_

#define EIGEN_USE_THREADS

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "machina/core/framework/op_kernel.h"
#include "machina/core/framework/tensor_types.h"

namespace machina {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;


enum DenseUpdateType { ADD, SUB, ASSIGN };

namespace functor {

template <typename Device, typename T, DenseUpdateType OP>
struct DenseUpdate {
  void operator()(const Device& d, typename TTypes<T>::Flat params,
                  typename TTypes<T>::ConstFlat update);
};

template <typename T>
struct DenseUpdate<CPUDevice, T, ADD> {
  void operator()(const CPUDevice& d, typename TTypes<T>::Flat params,
                  typename TTypes<T>::ConstFlat update) {
    params.device(d) += update;
  }
};

template <typename T>
struct DenseUpdate<CPUDevice, T, SUB> {
  void operator()(const CPUDevice& d, typename TTypes<T>::Flat params,
                  typename TTypes<T>::ConstFlat update) {
    params.device(d) -= update;
  }
};

template <typename T>
struct DenseUpdate<CPUDevice, T, ASSIGN> {
  void operator()(const CPUDevice& d, typename TTypes<T>::Flat params,
                  typename TTypes<T>::ConstFlat update) {
    params.device(d) = update;
  }
};


}  // end namespace functor

template <typename Device>
absl::Status VariantCopyFn(OpKernelContext* context, const Tensor& from,
                           Tensor* to);

template <>
absl::Status VariantCopyFn<CPUDevice>(OpKernelContext* context,
                                      const Tensor& from, Tensor* to);
template <>
absl::Status VariantCopyFn<GPUDevice>(OpKernelContext* context,
                                      const Tensor& from, Tensor* to);

}  // end namespace machina

#endif  // MACHINA_CORE_KERNELS_DENSE_UPDATE_FUNCTOR_H_
