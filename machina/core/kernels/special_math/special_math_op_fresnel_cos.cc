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

#include "machina/core/kernels/cwise_ops_common.h"
#include "machina/core/kernels/special_math/special_math_op_misc_impl.h"

namespace machina {
REGISTER2(UnaryOp, CPU, "FresnelCos", functor::fresnel_cos, float, double);
#if GOOGLE_CUDA || MACHINA_USE_ROCM
REGISTER2(UnaryOp, GPU, "FresnelCos", functor::fresnel_cos, float, double);
#endif  // GOOGLE_CUDA || MACHINA_USE_ROCM
}  // namespace machina
