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
#include "machina/c/experimental/ops/gen/cpp/renderers/op_comment_renderer.h"

#include "machina/c/experimental/ops/gen/cpp/renderers/renderer.h"
#include "machina/c/experimental/ops/gen/cpp/renderers/renderer_context.h"
#include "machina/c/experimental/ops/gen/cpp/views/op_view.h"

namespace machina {
namespace generator {
namespace cpp {

OpCommentRenderer::OpCommentRenderer(RendererContext context, OpView op)
    : Renderer(context), op_(op) {}

void OpCommentRenderer::Render() {
  if (context_.mode == RendererContext::kHeader) {
    // Add a short 1-line comment to the header files.
    CommentLine(op_.Summary());
    return;
  }

  CommentLine("Op: $0()", op_.FunctionName());
  CommentLine("Summary: $0", op_.Summary());
  CommentLine("");
  CommentLine("Description:");
  for (const auto& line : op_.Description()) {
    CommentLine("  $0", line);
  }
}

}  // namespace cpp
}  // namespace generator
}  // namespace machina
