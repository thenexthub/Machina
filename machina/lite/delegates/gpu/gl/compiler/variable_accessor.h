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

#ifndef MACHINA_LITE_DELEGATES_GPU_GL_COMPILER_VARIABLE_ACCESSOR_H_
#define MACHINA_LITE_DELEGATES_GPU_GL_COMPILER_VARIABLE_ACCESSOR_H_

#include <set>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "machina/lite/delegates/gpu/gl/compiler/preprocessor.h"
#include "machina/lite/delegates/gpu/gl/variable.h"

namespace tflite {
namespace gpu {
namespace gl {

// This rewrite handles access to variables. It may rewrite a variable with
// actual values if 'inline_values' is set to true.
//
// The following syntax is supported to access variables:
//  - simple variable: name
//  - variable with field: name.(x|y|z|w)
//  - variable with index: name[i]
//  - variable with index and field: name[i].(x|y|z|w)
//
// If 'inline_values' is set to true, non-variable-length variables will be
// inlined. For example, 'base.x' will be replaced with value of 'x' field from
// 'base'. Variable-length variables are declared as const and accessed via
// index. These declarations are returned by GetConstDeclarations.
//
// If 'inline_values' is set to false, all variables will be declared as
// uniforms. Uniform declarations are returned by GetUniformDeclarations.
class VariableAccessor : public InlineRewrite {
 public:
  explicit VariableAccessor(bool inline_values, bool vulkan_support = false)
      : inline_values_(inline_values), vulkan_support_(vulkan_support) {}

  RewriteStatus Rewrite(absl::string_view input, std::string* output) final;

  // Returns true if variable was successfully added.
  bool AddSharedVariable(Variable&& variable);

  // Returns true if variable was successfully added.
  bool AddUniformParameter(Variable&& variable);

  // Returns true if variable value is an empty vector.
  bool IsEmptyVariableLength(const Variable& variable) const;

  // Returns const variables that need to be inlined in the a shader's code.
  std::string GetConstDeclarations() const;

  // Returns shared variable declarations that need to be inlined.
  std::string GetSharedVariableDeclarations() const;

  // Returns uniform parameter declarations that need to be inlined.
  std::string GetUniformParameterDeclarations() const;

  // Returns a collection of uniform parameters.
  std::vector<Variable> GetUniformParameters() const;

 private:
  const bool inline_values_;
  const bool vulkan_support_;
  absl::flat_hash_map<std::string, Variable> name_to_variable_;
  std::set<std::string> shared_variables_;
  std::set<std::string> uniform_parameters_;
};

// Implementation details below.

namespace variable_accessor_internal {

struct VariableReference {
  absl::string_view name;
  absl::string_view index;
  absl::string_view field;
};

// Parse the following regex manually
// name(\[index\])?(\.field)?
VariableReference Parse(absl::string_view input);

}  // namespace variable_accessor_internal
}  // namespace gl
}  // namespace gpu
}  // namespace tflite

#endif  // MACHINA_LITE_DELEGATES_GPU_GL_COMPILER_VARIABLE_ACCESSOR_H_
