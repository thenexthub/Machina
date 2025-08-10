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

#ifndef MACHINA_LITE_DELEGATES_GPU_GL_OBJECT_MANAGER_H_
#define MACHINA_LITE_DELEGATES_GPU_GL_OBJECT_MANAGER_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "machina/lite/delegates/gpu/common/model.h"
#include "machina/lite/delegates/gpu/common/status.h"
#include "machina/lite/delegates/gpu/gl/gl_buffer.h"
#include "machina/lite/delegates/gpu/gl/gl_texture.h"
#include "machina/lite/delegates/gpu/gl/stats.h"

namespace tflite {
namespace gpu {
namespace gl {

// ObjectManager is a registry that owns corresponding objects and provides
// discovery functionality. All objects are kept until manager is destroyed.
//
// All buffers and textures share the same id space, therefore, it is an error
// to register two objects with the same id.
// TODO(akulik): make ObjectManager templated by object type.
class ObjectManager {
 public:
  // Moves ownership over the given buffer to the manager.
  absl::Status RegisterBuffer(uint32_t id, GlBuffer buffer);

  void RemoveBuffer(uint32_t id);

  // Return a permanent pointer to a buffer for the given id or nullptr.
  GlBuffer* FindBuffer(uint32_t id) const;

  // Moves ownership over the given texture to the manager.
  absl::Status RegisterTexture(uint32_t id, GlTexture texture);

  void RemoveTexture(uint32_t id);

  // Return a permanent pointer to a texture for the given id or nullptr.
  GlTexture* FindTexture(uint32_t id) const;

  ObjectsStats stats() const;

 private:
  std::vector<std::unique_ptr<GlBuffer>> buffers_;
  std::vector<std::unique_ptr<GlTexture>> textures_;
};

// TODO(akulik): find better place for functions below.

// Creates read-only buffer from the given tensor. Tensor data is converted to
// PHWC4 layout.
absl::Status CreatePHWC4BufferFromTensor(const TensorFloat32& tensor,
                                         GlBuffer* gl_buffer);

// Creates read-write buffer for the given tensor shape, where data layout is
// supposed to be PHWC4.
absl::Status CreatePHWC4BufferFromTensorRef(const TensorRef<BHWC>& tensor_ref,
                                            GlBuffer* gl_buffer);

// Copies data from a buffer that holds data in PHWC4 layout to the given
// tensor.
absl::Status CopyFromPHWC4Buffer(const GlBuffer& buffer, TensorFloat32* tensor);

}  // namespace gl
}  // namespace gpu
}  // namespace tflite

#endif  // MACHINA_LITE_DELEGATES_GPU_GL_OBJECT_MANAGER_H_
