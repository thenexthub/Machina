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

#ifndef MACHINA_LITE_DELEGATES_GPU_GL_COMPILER_SHADER_CODEGEN_H_
#define MACHINA_LITE_DELEGATES_GPU_GL_COMPILER_SHADER_CODEGEN_H_

#include <string>
#include <vector>

#include "machina/lite/delegates/gpu/common/gpu_info.h"
#include "machina/lite/delegates/gpu/common/model.h"
#include "machina/lite/delegates/gpu/common/status.h"
#include "machina/lite/delegates/gpu/gl/compiler/compiled_node.h"
#include "machina/lite/delegates/gpu/gl/compiler/object_accessor.h"
#include "machina/lite/delegates/gpu/gl/compiler/shader_code.h"
#include "machina/lite/delegates/gpu/gl/compiler_options.h"
#include "machina/lite/delegates/gpu/gl/object.h"

namespace tflite {
namespace gpu {
namespace gl {

// This class is responsible for assembling a shader by putting together
// objects, parameters declarations and main function.
class ShaderCodegen {
 public:
  ShaderCodegen(const CompilationOptions& options, const GpuInfo& gpu_info);

  // Builds final program representation.
  absl::Status Build(CompiledNodeAttributes attr,
                     ShaderCode* shader_code) const;

 private:
  const CompilationOptions options_;
  const GpuVendor gpu_type_;
  bool inline_parameters_;
};

}  // namespace gl
}  // namespace gpu
}  // namespace tflite

#endif  // MACHINA_LITE_DELEGATES_GPU_GL_COMPILER_SHADER_CODEGEN_H_
