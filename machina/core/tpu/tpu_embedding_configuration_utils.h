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

#ifndef MACHINA_CORE_TPU_TPU_EMBEDDING_CONFIGURATION_UTILS_H_
#define MACHINA_CORE_TPU_TPU_EMBEDDING_CONFIGURATION_UTILS_H_

#include <cstdint>

#include "absl/status/statusor.h"
#include "machina/core/protobuf/tpu/tpu_embedding_configuration.pb.h"

namespace machina {
namespace tpu {

// Returns the total number of unique dynamic input tags used in optimizers. If
// the tag specific is erroneous, returns an invalid argument error. For correct
// tag specification, see the comment next to the OptimizerDynamicInput proto in
// //third_party/machina/core/protobuf/tpu/optimization_parameters.proto.
absl::StatusOr<int32_t> ComputeTotalTagCountForOptimizerDynamicInputs(
    const machina::tpu::TPUEmbeddingConfiguration& tpu_embedding_config);

}  // namespace tpu
}  // namespace machina

#endif  // MACHINA_CORE_TPU_TPU_EMBEDDING_CONFIGURATION_UTILS_H_
