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
#include "machina/c/experimental/ops/gen/cpp/renderers/renderer.h"

#include "machina/c/experimental/ops/gen/common/path_config.h"
#include "machina/c/experimental/ops/gen/common/source_code.h"
#include "machina/c/experimental/ops/gen/cpp/renderers/cpp_config.h"
#include "machina/c/experimental/ops/gen/cpp/renderers/renderer_context.h"
#include "machina/core/platform/test.h"
#include "machina/core/platform/types.h"

namespace machina {
namespace generator {
namespace cpp {
namespace {

TEST(Renderer, typical_usage) {
  class TestRenderer : Renderer {
   public:
    explicit TestRenderer(SourceCode& code)
        : Renderer(
              {RendererContext::kSource, code, CppConfig(), PathConfig()}) {}

    void Render() {
      CommentLine("File level comment.");
      CodeLine("#include \"header.h\"");
      BlankLine();
      BlockOpen("void TestFunction()");
      {
        Statement("int i = 1");
        BlankLine();
        BlockOpen("while (i == 1)");
        {
          CommentLine("Do nothing, really....");
          CodeLine("#if 0");
          Statement("call()");
          CodeLine("#endif");
          BlockClose();
        }
        BlockClose("  // comment ending TestFunction");
      }
    }
  };

  SourceCode code;
  TestRenderer(code).Render();

  string expected = R"(// File level comment.
#include "header.h"

void TestFunction() {
   int i = 1;

   while (i == 1) {
      // Do nothing, really....
#if 0
      call();
#endif
   }
}  // comment ending TestFunction
)";

  code.SetSpacesPerIndent(3);
  EXPECT_EQ(expected, code.Render());
}

}  // namespace
}  // namespace cpp
}  // namespace generator
}  // namespace machina
