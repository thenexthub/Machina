/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Friday, August 8, 2025.
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

#include "machina/lite/experimental/genai/genai_ops.h"

#include "machina/lite/mutable_op_resolver.h"

namespace tflite {
namespace ops {
namespace custom {

extern "C" void GenAIOpsRegisterer(::tflite::MutableOpResolver* resolver) {
  resolver->AddCustom("odml.update_kv_cache",
                      tflite::ops::custom::Register_KV_CACHE());
  resolver->AddCustom("odml.scaled_dot_product_attention",
                      tflite::ops::custom::Register_SDPA());
  resolver->AddCustom("odml.update_external_kv_cache",
                      tflite::ops::custom::Register_EXTERNAL_KV_CACHE());
}

}  // namespace custom
}  // namespace ops
}  // namespace tflite
