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

#ifndef MACHINA_COMPILER_TF2MACHINA_XLAKERNELS_RNG_CONVERTER_UTILS_H_
#define MACHINA_COMPILER_TF2MACHINA_XLAKERNELS_RNG_CONVERTER_UTILS_H_

#include "absl/strings/string_view.h"
#include "machina/xla/xla_data.pb.h"
#include "machina/core/framework/rng_alg.h"

namespace machina {

// Given the XLA::RandomAlgorithm, return the Tensorflow equivalent.
Algorithm ToTensorflowAlgorithm(xla::RandomAlgorithm alg);

// Given the device type, return the default XLA::RandomAlgorithm
xla::RandomAlgorithm DefaultRngAlgForDeviceType(
    absl::string_view device_type_string);

}  // namespace machina

#endif  // MACHINA_COMPILER_TF2MACHINA_XLAKERNELS_RNG_CONVERTER_UTILS_H_
