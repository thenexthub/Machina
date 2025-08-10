/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, May 15, 2025.
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

// This include can't be in the conv_ops_fused_impl.h headers. See b/62899350.
#if GOOGLE_CUDA
#include "machina/core/protobuf/autotuning.pb.h"
#endif  // GOOGLE_CUDA
#include "machina/core/kernels/conv_ops_fused_impl.h"

namespace machina {

// If we're using the alternative GEMM-based implementation of Conv2D for the
// CPU implementation, don't register this EigenTensor-based version.
#if !defined(USE_GEMM_FOR_CONV)
TF_CALL_float(REGISTER_FUSED_CPU_CONV2D);
#endif  // !USE_GEMM_FOR_CONV

#if GOOGLE_CUDA

namespace functor {
DECLARE_FUNCTOR_GPU_SPEC(float);
}  // namespace functor

TF_CALL_float(REGISTER_FUSED_GPU_CONV2D);

#endif  // GOOGLE_CUDA

}  // namespace machina
