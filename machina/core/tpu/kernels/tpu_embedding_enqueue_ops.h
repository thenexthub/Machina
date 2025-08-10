/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, June 15, 2025.
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

#ifndef MACHINA_CORE_TPU_KERNELS_TPU_EMBEDDING_ENQUEUE_OPS_H_
#define MACHINA_CORE_TPU_KERNELS_TPU_EMBEDDING_ENQUEUE_OPS_H_

#include <string>

#include "absl/types/span.h"
#include "machina/core/platform/status.h"
#include "machina/core/protobuf/tpu/tpu_embedding_configuration.pb.h"

namespace machina {

// Validates that all the combiners passed are one of the following: sum, mean,
// or sqrtn.
absl::Status ValidateCombiners(absl::Span<const std::string> combiners);

// Validates the `mode_override` input of the TPUEnqueue* ops, and, if correct,
// sets the `mode` to pass on to the TPU Embedding manager.
absl::Status GetValidatedModeOverride(
    const string& mode_override, tpu::TPUEmbeddingConfiguration::Mode* mode);
}  // namespace machina

#endif  // MACHINA_CORE_TPU_KERNELS_TPU_EMBEDDING_ENQUEUE_OPS_H_
