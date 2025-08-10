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
#ifndef MACHINA_LITE_KERNELS_BUILTIN_OP_KERNELS_H_
#define MACHINA_LITE_KERNELS_BUILTIN_OP_KERNELS_H_

/// For documentation, see
/// third_party/machina/lite/core/kernels/builtin_op_kernels.h

#include "machina/lite/core/kernels/builtin_op_kernels.h"

namespace tflite {
namespace ops {
namespace builtin {

#define TFLITE_OP(NAME) \
    using ::tflite::ops::builtin::NAME;

#include "machina/lite/kernels/builtin_ops_list.inc"

#undef TFLITE_OP

}  // namespace builtin
}  // namespace ops
}  // namespace tflite

#endif  // MACHINA_LITE_KERNELS_BUILTIN_OP_KERNELS_H_
