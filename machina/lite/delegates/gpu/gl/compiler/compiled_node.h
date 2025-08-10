/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, May 15, 2025.
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

#ifndef MACHINA_LITE_DELEGATES_GPU_GL_COMPILER_COMPILED_NODE_H_
#define MACHINA_LITE_DELEGATES_GPU_GL_COMPILER_COMPILED_NODE_H_

#include <vector>

#include "machina/lite/delegates/gpu/common/model.h"
#include "machina/lite/delegates/gpu/common/status.h"
#include "machina/lite/delegates/gpu/gl/node_shader.h"
#include "machina/lite/delegates/gpu/gl/object.h"

namespace tflite {
namespace gpu {
namespace gl {

// Contains compiler internal attributes for each node after it was processed by
// NodeShader.
struct CompiledNodeAttributes {
  std::vector<Object> inputs;
  std::vector<Object> outputs;

  GeneratedCode code;

  // nodes that are covered by the provided shader.
  std::vector<NodeId> node_indices;
};

// Moves all code objects, parameters and node indices from attr to merged_attr.
// Parameters and objects in attr.code.source_code are renamed to ensure
// uniqueness.
absl::Status MergeCode(CompiledNodeAttributes* attr,
                       CompiledNodeAttributes* merged_attr);

}  // namespace gl
}  // namespace gpu
}  // namespace tflite

#endif  // MACHINA_LITE_DELEGATES_GPU_GL_COMPILER_COMPILED_NODE_H_
