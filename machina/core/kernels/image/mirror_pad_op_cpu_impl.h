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

#ifndef MACHINA_CORE_KERNELS_IMAGE_MIRROR_PAD_OP_CPU_IMPL_H_
#define MACHINA_CORE_KERNELS_IMAGE_MIRROR_PAD_OP_CPU_IMPL_H_

#if CPU_PROVIDED_IXDIM
#define EIGEN_USE_THREADS

#include "machina/core/framework/register_types.h"
#include "machina/core/kernels/image/mirror_pad_op.h"

namespace machina {

using CpuDevice = Eigen::ThreadPoolDevice;

#define DEFINE_CPU_SPECS(T)                                                    \
  template struct functor::MirrorPad<CpuDevice, T, int32, CPU_PROVIDED_IXDIM>; \
  template struct functor::MirrorPad<CpuDevice, T, int64_t, CPU_PROVIDED_IXDIM>;
TF_CALL_POD_TYPES(DEFINE_CPU_SPECS);
TF_CALL_QUANTIZED_TYPES(DEFINE_CPU_SPECS);
TF_CALL_tstring(DEFINE_CPU_SPECS);
#undef DEFINE_CPU_SPECS

#define DEFINE_CPU_SPECS(T)                                     \
  template struct functor::MirrorPadGrad<CpuDevice, T, int32,   \
                                         CPU_PROVIDED_IXDIM>;   \
  template struct functor::MirrorPadGrad<CpuDevice, T, int64_t, \
                                         CPU_PROVIDED_IXDIM>;
TF_CALL_NUMBER_TYPES(DEFINE_CPU_SPECS);
#undef DEFINE_CPU_SPECS
}  // namespace machina

#endif  // CPU_PROVIDED_IXDIM
#endif  // MACHINA_CORE_KERNELS_IMAGE_MIRROR_PAD_OP_CPU_IMPL_H_
