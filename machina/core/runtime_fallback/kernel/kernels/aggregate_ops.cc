/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Saturday, June 21, 2025.
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

// Register TensorFlow's AddNOp so it can be called directly from TFRT.

#define EIGEN_USE_THREADS

#include "machina/core/kernels/aggregate_ops.h"

#include "machina/core/kernels/aggregate_ops_cpu.h"
#include "machina/core/runtime_fallback/kernel/tfrt_op_kernel.h"

namespace machina {

// TODO(lauj) Share op properties with TF's op registration in math_ops.cc. This
// requires supporting attribute outputs ("sum: T") and compound attribute types
// ("T: {numbertype, variant}").
REGISTER_KERNEL_FALLBACK_OP("AddN").Output("out: int32");

REGISTER_KERNEL_FALLBACK_KERNEL(
    "AddN", AddNOp<CPUDevice, int32, TFRTOpKernel, TFRTOpKernelConstruction,
                   TFRTOpKernelContext>);

}  // namespace machina
