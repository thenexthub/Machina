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
#include "machina/c/experimental/ops/gen/cpp/renderers/include_renderer.h"

#include "machina/c/experimental/ops/gen/cpp/renderers/renderer.h"
#include "machina/c/experimental/ops/gen/cpp/renderers/renderer_context.h"
#include "machina/core/platform/path.h"
#include "machina/core/platform/types.h"

namespace machina {
namespace generator {
namespace cpp {

IncludeRenderer::IncludeRenderer(RendererContext context) : Renderer(context) {}

void IncludeRenderer::SelfHeader() {
  Include(SelfHeaderPath());
  BlankLine();
}

string IncludeRenderer::SelfHeaderPath() const {
  return io::JoinPath(context_.path_config.tf_root_dir,
                      context_.path_config.tf_output_dir,
                      context_.cpp_config.unit + "_ops.h");
}

void IncludeRenderer::Include(const string &tf_file_path) {
  CodeLine("#include \"$0\"",
           io::JoinPath(context_.path_config.tf_prefix_dir, tf_file_path));
}

void IncludeRenderer::Headers() {
  Include("machina/c/eager/abstract_context.h");
  Include("machina/c/eager/abstract_tensor_handle.h");
  if (context_.mode == RendererContext::kSource) {
    Include("machina/c/eager/tracing_utils.h");
    Include("machina/core/framework/types.h");
    Include("machina/core/platform/errors.h");
    BlankLine();
    Statement("using machina::tracing::MaybeSetOpName");
  }
  BlankLine();
}

}  // namespace cpp
}  // namespace generator
}  // namespace machina
