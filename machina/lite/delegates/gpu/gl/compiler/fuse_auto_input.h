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

#ifndef MACHINA_LITE_DELEGATES_GPU_GL_COMPILER_FUSE_AUTO_INPUT_H_
#define MACHINA_LITE_DELEGATES_GPU_GL_COMPILER_FUSE_AUTO_INPUT_H_

#include "machina/lite/delegates/gpu/common/model.h"
#include "machina/lite/delegates/gpu/common/model_transformer.h"

namespace tflite {
namespace gpu {
namespace gl {

// Fuses nodes that have auto output with auto input node using the following
// rules.
//
// Source graph:
//   A B C
//   \ | /
//     D
//
// - A, B and C each have a single output marked as AUTO
// - Each output is used only by D
// - D has all inputs marked as AUTO
//
// Result: in the best case a single node that does (A,B,C)+D operations.
//
class FuseAutoInput : public NodeTransformation {
 public:
  TransformResult ApplyToNode(Node* node, GraphFloat32* graph) final;
};

}  // namespace gl
}  // namespace gpu
}  // namespace tflite

#endif  // MACHINA_LITE_DELEGATES_GPU_GL_COMPILER_FUSE_AUTO_INPUT_H_
