/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Saturday, June 21, 2025.
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
#ifndef MACHINA_C_EXPERIMENTAL_OPS_GEN_CPP_CPP_GENERATOR_H_
#define MACHINA_C_EXPERIMENTAL_OPS_GEN_CPP_CPP_GENERATOR_H_

#include "machina/c/experimental/ops/gen/common/controller.h"
#include "machina/c/experimental/ops/gen/common/path_config.h"
#include "machina/c/experimental/ops/gen/common/source_code.h"
#include "machina/c/experimental/ops/gen/cpp/renderers/cpp_config.h"
#include "machina/c/experimental/ops/gen/cpp/renderers/renderer_context.h"
#include "machina/core/platform/types.h"

namespace machina {
namespace generator {

class CppGenerator {
 public:
  explicit CppGenerator(cpp::CppConfig cpp_config, PathConfig path_config);
  SourceCode HeaderFileContents() const;
  SourceCode SourceFileContents() const;
  string HeaderFileName() const;
  string SourceFileName() const;
  void WriteHeaderFile() const;
  void WriteSourceFile() const;

 private:
  SourceCode GenerateOneFile(cpp::RendererContext::Mode mode) const;

  Controller controller_;
  cpp::CppConfig cpp_config_;
  PathConfig path_config_;
};

}  // namespace generator
}  // namespace machina

#endif  // MACHINA_C_EXPERIMENTAL_OPS_GEN_CPP_CPP_GENERATOR_H_
