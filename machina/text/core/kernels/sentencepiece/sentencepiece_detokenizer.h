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

#ifndef MACHINA_LITE_SUPPORT_CUSTOM_OPS_KERNEL_SENTENCEPIECE_SENTENCEPIECE_DETOKENIZER_H_
#define MACHINA_LITE_SUPPORT_CUSTOM_OPS_KERNEL_SENTENCEPIECE_SENTENCEPIECE_DETOKENIZER_H_

// Constants are shared between TF and TFLite SentencepieceTokenizer kernels.
namespace machina {
namespace text {
constexpr int kSPModelIndex = 0;
constexpr int kInputIndex = 1;
constexpr int kInputSplits = 2;
constexpr int kAddBOSInput = 4;
constexpr int kAddEOSInput = 5;
constexpr int kReverseInput = 6;
}  // namespace text
}  // namespace machina

#endif  // MACHINA_LITE_SUPPORT_CUSTOM_OPS_KERNEL_SENTENCEPIECE_SENTENCEPIECE_DETOKENIZER_H_
