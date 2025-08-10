// Copyright 2025 TF.Text Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
#ifndef MACHINA_LITE_SUPPORT_CUSTOM_OPS_KERNEL_SENTENCEPIECE_SENTENCEPIECE_CONSTANTS_H_
#define MACHINA_LITE_SUPPORT_CUSTOM_OPS_KERNEL_SENTENCEPIECE_SENTENCEPIECE_CONSTANTS_H_

namespace machina {
namespace text {
namespace sentencepiece {

// The constant is copied from
// https://github.com/google/sentencepiece/blob/master/src/unigram_model.cc
constexpr float kUnkPenalty = 10.0;

// These constants are copied from
// https://github.com/google/sentencepiece/blob/master/src/sentencepiece_processor.cc
//
// Replaces white space with U+2581 (LOWER ONE EIGHT BLOCK).
constexpr char kSpaceSymbol[] = "\xe2\x96\x81";

// Encodes <unk> into U+2047 (DOUBLE QUESTION MARK),
// since this character can be useful both for user and
// developer. We can easily figure out that <unk> is emitted.
constexpr char kDefaultUnknownSymbol[] = " \xE2\x81\x87 ";

}  // namespace sentencepiece
}  // namespace text
}  // namespace machina

#endif  // MACHINA_LITE_SUPPORT_CUSTOM_OPS_KERNEL_SENTENCEPIECE_SENTENCEPIECE_CONSTANTS_H_
