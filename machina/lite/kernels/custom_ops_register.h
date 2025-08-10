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
#ifndef MACHINA_LITE_KERNELS_CUSTOM_OPS_REGISTER_H_
#define MACHINA_LITE_KERNELS_CUSTOM_OPS_REGISTER_H_

#include "machina/lite/core/c/common.h"

namespace tflite {
namespace ops {
namespace custom {

TfLiteRegistration* Register_ATAN2();
TfLiteRegistration* Register_AVG_POOL_3D();
TfLiteRegistration* Register_HASHTABLE();
TfLiteRegistration* Register_HASHTABLE_FIND();
TfLiteRegistration* Register_HASHTABLE_IMPORT();
TfLiteRegistration* Register_HASHTABLE_SIZE();
TfLiteRegistration* Register_IRFFT2D();
TfLiteRegistration* Register_MAX_POOL_3D();
TfLiteRegistration* Register_MULTINOMIAL();
TfLiteRegistration* Register_RANDOM_STANDARD_NORMAL();
TfLiteRegistration* Register_RANDOM_UNIFORM();
TfLiteRegistration* Register_RANDOM_UNIFORM_INT();
TfLiteRegistration* Register_ROLL();
TfLiteRegistration* Register_SIGN();
TfLiteRegistration* Register_TABLE();

}  // namespace custom
}  // namespace ops
}  // namespace tflite

#endif  // MACHINA_LITE_KERNELS_CUSTOM_OPS_REGISTER_H_
