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

#ifndef MACHINA_LITE_DELEGATES_GPU_GL_KERNELS_TEST_UTIL_H_
#define MACHINA_LITE_DELEGATES_GPU_GL_KERNELS_TEST_UTIL_H_

#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "machina/lite/delegates/gpu/common/model.h"
#include "machina/lite/delegates/gpu/common/operations.h"
#include "machina/lite/delegates/gpu/common/status.h"
#include "machina/lite/delegates/gpu/common/tensor.h"
#include "machina/lite/delegates/gpu/gl/compiler_options.h"
#include "machina/lite/delegates/gpu/gl/node_shader.h"
#include "machina/lite/delegates/gpu/gl/runtime_options.h"

#ifndef ASSERT_OK
#define ASSERT_OK(x) ASSERT_THAT(x.message(), testing::StrEq(""));
#endif

namespace tflite {
namespace gpu {
namespace gl {

class SingleOpModel {
 public:
  SingleOpModel() = delete;
  SingleOpModel(Operation&& operation,
                const std::vector<TensorRef<BHWC>>& inputs,
                const std::vector<TensorRef<BHWC>>& outputs);

  virtual ~SingleOpModel() = default;

  bool PopulateTensor(int index, std::vector<float>&& data);

  absl::Status Invoke(const NodeShader& shader);
  absl::Status Invoke(const CompilationOptions& compile_options,
                      const RuntimeOptions& runtime_options,
                      const NodeShader& shader);

  const std::vector<float>& GetOutput(int index) const {
    return outputs_[index].data;
  }

 protected:
  GraphFloat32 graph_;
  std::vector<TensorFloat32> inputs_;
  std::vector<TensorFloat32> outputs_;
};

}  // namespace gl
}  // namespace gpu
}  // namespace tflite

#endif  // MACHINA_LITE_DELEGATES_GPU_GL_KERNELS_TEST_UTIL_H_
