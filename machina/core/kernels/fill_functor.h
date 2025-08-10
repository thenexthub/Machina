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

#ifndef MACHINA_CORE_KERNELS_FILL_FUNCTOR_H_
#define MACHINA_CORE_KERNELS_FILL_FUNCTOR_H_

#define EIGEN_USE_THREADS

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "machina/core/framework/tensor_types.h"
#include "machina/core/framework/types.h"

namespace machina {
namespace functor {

template <typename Device, typename T>
struct FillFunctor {
  // Computes on device "d": out = out.constant(in(0)),
  void operator()(const Device& d, typename TTypes<T>::Flat out,
                  typename TTypes<T>::ConstScalar in);
};

template <typename Device, typename T>
struct SetZeroFunctor {
  // Computes on device "d": out = out.setZero(),
  void operator()(const Device& d, typename TTypes<T>::Flat out);
};

// Partial specialization of SetZeroFunctor<Device=Eigen::ThreadPoolDevice, T>.
template <typename T>
struct SetZeroFunctor<Eigen::ThreadPoolDevice, T> {
  void operator()(const Eigen::ThreadPoolDevice& d,
                  typename TTypes<T>::Flat out);
};


template <>
struct SetZeroFunctor<Eigen::ThreadPoolDevice, tstring> {
  void operator()(const Eigen::ThreadPoolDevice& d,
                  typename TTypes<tstring>::Flat out);
};

template <typename Device, typename T>
struct SetOneFunctor {
  // Computes on device "d": out = out.setOne(),
  void operator()(const Device& d, typename TTypes<T>::Flat out);
};

// Partial specialization of SetOneFunctor<Device=Eigen::ThreadPoolDevice, T>.
template <typename T>
struct SetOneFunctor<Eigen::ThreadPoolDevice, T> {
  void operator()(const Eigen::ThreadPoolDevice& d,
                  typename TTypes<T>::Flat out);
};


template <>
struct SetOneFunctor<Eigen::ThreadPoolDevice, tstring> {
  void operator()(const Eigen::ThreadPoolDevice& d,
                  typename TTypes<tstring>::Flat out);
};

template <typename Device, typename T>
struct SetNanFunctor {
  void operator()(const Device& d, typename TTypes<T>::Flat out);
};

// Partial specialization of SetNanFunctor<Device=Eigen::ThreadPoolDevice, T>.
template <typename T>
struct SetNanFunctor<Eigen::ThreadPoolDevice, T> {
  void operator()(const Eigen::ThreadPoolDevice& d,
                  typename TTypes<T>::Flat out);
};

template <>
struct SetNanFunctor<Eigen::ThreadPoolDevice, tstring> {
  void operator()(const Eigen::ThreadPoolDevice& d,
                  typename TTypes<tstring>::Flat out);
};

}  // namespace functor
}  // namespace machina

#endif  // MACHINA_CORE_KERNELS_FILL_FUNCTOR_H_
