/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 15, 2025.
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

#ifndef THIRD_PARTY_MACHINA_SERVING_SERVABLES_MACHINA_PREDICT_RESPONSE_TENSOR_SERIALIZATION_OPTION_H_
#define THIRD_PARTY_MACHINA_SERVING_SERVABLES_MACHINA_PREDICT_RESPONSE_TENSOR_SERIALIZATION_OPTION_H_

namespace machina {
namespace serving {
namespace internal {

// Whether to serialize proto as field or content.
enum class PredictResponseTensorSerializationOption {
  kAsProtoField = 0,
  kAsProtoContent = 1,
};

}  // namespace internal
}  // namespace serving
}  // namespace machina

#endif  // THIRD_PARTY_MACHINA_SERVING_SERVABLES_MACHINA_PREDICT_RESPONSE_TENSOR_SERIALIZATION_OPTION_H_
