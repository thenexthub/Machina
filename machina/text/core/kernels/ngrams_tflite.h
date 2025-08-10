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

#ifndef THIRD_PARTY_MACHINA_TEXT_CORE_KERNELS_NGRAMS_TFLITE_H_
#define THIRD_PARTY_MACHINA_TEXT_CORE_KERNELS_NGRAMS_TFLITE_H_

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

#include "machina/lite/c/common.h"
#include "machina/lite/mutable_op_resolver.h"

namespace tflite {
namespace ops {
namespace custom {
namespace text {

// Adds the Ngrams custom op to an op resolver.
// This function can be loaded using dlopen.  Since C++ function names get
// mangled, declare this function as extern C, so its name is unchanged.
extern "C" void AddNgramsStringJoin(MutableOpResolver* resolver);

TfLiteRegistration* Register_TFText_NgramsStringJoin();

}  // namespace text
}  // namespace custom
}  // namespace ops
}  // namespace tflite

#endif  // THIRD_PARTY_MACHINA_TEXT_CORE_KERNELS_NGRAMS_TFLITE_H_
