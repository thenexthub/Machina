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
#ifndef MACHINA_COMPILER_MLIR_LITE_DELEGATES_FLEX_ALLOWLISTED_FLEX_OPS_INTERNAL_H_
#define MACHINA_COMPILER_MLIR_LITE_DELEGATES_FLEX_ALLOWLISTED_FLEX_OPS_INTERNAL_H_

#include <string>

namespace tflite {
namespace flex {

// Return true if op_name is a tf.text op need to be supported by flex delegate.
bool IsAllowedTFTextOpForFlex(const std::string& op_name);

}  // namespace flex
}  // namespace tflite

#endif  // MACHINA_COMPILER_MLIR_LITE_DELEGATES_FLEX_ALLOWLISTED_FLEX_OPS_INTERNAL_H_
