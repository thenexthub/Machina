/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, July 12, 2025.
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
#ifndef MACHINA_LITE_KERNELS_INTERNAL_OPAQUE_TENSOR_CTYPES_H_
#define MACHINA_LITE_KERNELS_INTERNAL_OPAQUE_TENSOR_CTYPES_H_

#include "machina/lite/c/c_api_opaque.h"
#include "machina/lite/core/macros.h"
#include "machina/lite/kernels/internal/types.h"
#include "machina/lite/namespace.h"

namespace tflite {
namespace TFLITE_CONDITIONAL_NAMESPACE {

/// Returns the dimensions of the given tensor.
TFLITE_NOINLINE RuntimeShape GetTensorShape(const TfLiteOpaqueTensor* tensor);

}  // namespace TFLITE_CONDITIONAL_NAMESPACE

using ::tflite::TFLITE_CONDITIONAL_NAMESPACE::GetTensorShape;

}  // namespace tflite

#endif  // MACHINA_LITE_KERNELS_INTERNAL_OPAQUE_TENSOR_CTYPES_H_
