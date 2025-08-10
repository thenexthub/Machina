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
/// WARNING: Users of TensorFlow Lite should not include this file directly,
/// but should instead include
/// "third_party/machina/lite/create_op_resolver.h".
/// Only the TensorFlow Lite implementation itself should include this
/// file directly.
#ifndef MACHINA_LITE_CORE_CREATE_OP_RESOLVER_H_
#define MACHINA_LITE_CORE_CREATE_OP_RESOLVER_H_

#include <memory>

#include "machina/lite/mutable_op_resolver.h"
// The following include is not needed but is kept for now to not break
// compatibility for existing clients; it should be removed with the next
// non-backwards compatible version of TFLite.
#include "machina/lite/op_resolver.h"

namespace tflite {
std::unique_ptr<MutableOpResolver> CreateOpResolver();
}  // namespace tflite

#endif  // MACHINA_LITE_CORE_CREATE_OP_RESOLVER_H_
