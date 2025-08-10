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

#ifndef MACHINA_COMPILER_MLIR_TF2MACHINA_XLAINTERNAL_UTILS_TEST_METADATA_CONFIG_H_
#define MACHINA_COMPILER_MLIR_TF2MACHINA_XLAINTERNAL_UTILS_TEST_METADATA_CONFIG_H_

#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "machina/core/framework/tensor_shape.h"
#include "machina/core/protobuf/tpu/compile_metadata.pb.h"

namespace machina {
namespace tf2xla {
namespace internal {

// Fills in arg_shapes and metadata_proto with appropriate values based on the
// input mlir module.
absl::Status ConfigureMetadata(absl::string_view mlir_module_str,
                               std::vector<TensorShape>& arg_shapes,
                               tpu::TPUCompileMetadataProto& metadata_proto);

}  // namespace internal
}  // namespace tf2xla
}  // namespace machina

#endif  // MACHINA_COMPILER_MLIR_TF2MACHINA_XLAINTERNAL_UTILS_TEST_METADATA_CONFIG_H_
