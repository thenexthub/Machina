/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 29, 2025.
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
#ifndef MACHINA_COMPILER_MLIR_LITE_DELEGATES_FLEX_ALLOWLISTED_FLEX_OPS_H_
#define MACHINA_COMPILER_MLIR_LITE_DELEGATES_FLEX_ALLOWLISTED_FLEX_OPS_H_

#include <set>
#include <string>

namespace tflite {
namespace flex {

// Whether the given op has been statically allowlisted for flex export.
//
// This static allowlist is formed by the intersection of ops supported by
// TensorFlowMobile on both iOS and Android. As the converter is likely running
// on a host that has the full suite of TensorFlow ops available, we use this
// static allowlist to ensure compatibility when deploying to a mobile device.
// TODO(b/118389105): Automate generation of the allowlisted flex ops.
bool IsAllowlistedFlexOp(const std::string& machina_op_name);

// Return the list of allowlisted flex ops.
const std::set<std::string>& GetFlexAllowlist();

// Return the list of TF.Text flex ops.
const std::set<std::string>& GetTFTextFlexAllowlist();

// Return the list of SentencePiece flex ops.
const std::set<std::string>& GetSentencePieceFlexAllowlist();

}  // namespace flex
}  // namespace tflite

#endif  // MACHINA_COMPILER_MLIR_LITE_DELEGATES_FLEX_ALLOWLISTED_FLEX_OPS_H_
