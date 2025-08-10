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

#include "absl/log/log.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "machina/c/experimental/ops/gen/cpp/renderers/renderer_context.h"
#include "machina/core/lib/strings/str_util.h"
#include "machina/core/platform/logging.h"
#include "machina/core/platform/stringpiece.h"

namespace machina {
namespace generator {
namespace cpp {

Renderer::Renderer(RendererContext context) : context_(context) {}

Renderer& Renderer::BlankLine() {
  context_.code.AddLineWithoutIndent("");
  return *this;
}

Renderer& Renderer::CodeLine(const string& text) {
  context_.code.AddLineWithoutIndent(text);
  return *this;
}

Renderer& Renderer::CodeLines(const string& text) {
  absl::string_view trimmed_text(text);
  str_util::RemoveWhitespaceContext(&trimmed_text);
  for (const string& line : str_util::Split(trimmed_text, '\n')) {
    context_.code.AddLineWithoutIndent(line);
  }
  return *this;
}

Renderer& Renderer::Statement(const string& text) {
  if (absl::EndsWith(text, ";")) {
    LOG(WARNING) << "Superfluous terminating ';' in '" << text << "'";
    context_.code.AddLineWithIndent(text);
  } else {
    context_.code.AddLineWithIndent(absl::StrCat(text, ";"));
  }
  return *this;
}

Renderer& Renderer::TFStatement(const string& text) {
  return Statement(absl::Substitute("TF_RETURN_IF_ERROR($0)", text));
}

Renderer& Renderer::CommentLine(const string& text) {
  context_.code.AddLineWithIndent(absl::StrCat("// ", text));
  return *this;
}

Renderer& Renderer::BlockOpen(const string& text) {
  context_.code.AddLineWithIndent(absl::StrCat(text, " {"));
  context_.code.IncreaseIndent();
  return *this;
}

Renderer& Renderer::BlockClose(const string& text) {
  context_.code.DecreaseIndent();
  context_.code.AddLineWithIndent(absl::StrCat("}", text));
  return *this;
}

}  // namespace cpp
}  // namespace generator
}  // namespace machina
