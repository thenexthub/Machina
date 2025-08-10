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

#ifndef MACHINA_LITE_DELEGATES_GPU_GL_GL_SHADER_H_
#define MACHINA_LITE_DELEGATES_GPU_GL_GL_SHADER_H_

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "machina/lite/delegates/gpu/common/status.h"
#include "machina/lite/delegates/gpu/gl/portable_gl31.h"

namespace tflite {
namespace gpu {
namespace gl {

// A wrapper around opengl shader id that needs to be recycled when not needed.
class GlShader {
 public:
  // Creates and compiles a shader.
  //
  // @param shader_type is one of GL_VERTEX_SHADER, GL_FRAGMENT_SHADER, or
  // GL_COMPUTE_SHADER.
  static absl::Status CompileShader(GLenum shader_type,
                                    const std::string& shader_source,
                                    GlShader* gl_shader);

  GlShader() : id_(0) {}

  // move-only
  GlShader(GlShader&& shader);
  GlShader& operator=(GlShader&& shader);
  GlShader(const GlShader&) = delete;
  GlShader& operator=(const GlShader&) = delete;

  ~GlShader();

  GLuint id() const { return id_; }

 private:
  explicit GlShader(GLuint id) : id_(id) {}

  void Invalidate();

  GLuint id_;
};

// Holds binary blob for compiled shader. It can be used to instantiate
// a program instead of plain Shader that will need to be compiled first.
//
// Some OpenGL implementations allow to extract binary representation once it
// is compiled. Call Program::GetBinary after program is successfully created
// with a shader from sources.
class BinaryShader {
 public:
  BinaryShader(GLenum format, std::vector<uint8_t> binary)
      : format_(format), binary_(std::move(binary)) {}

  GLenum format() const { return format_; }

  const std::vector<uint8_t>& binary() const { return binary_; }

 private:
  GLenum format_;
  std::vector<uint8_t> binary_;
};

}  // namespace gl
}  // namespace gpu
}  // namespace tflite

#endif  // MACHINA_LITE_DELEGATES_GPU_GL_GL_SHADER_H_
