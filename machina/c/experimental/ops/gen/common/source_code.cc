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
#include "machina/c/experimental/ops/gen/common/source_code.h"

#include "absl/log/log.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/strip.h"
#include "machina/core/platform/logging.h"
#include "machina/core/platform/stringpiece.h"

namespace machina {
namespace generator {

string SourceCode::Render() const {
  string code;
  for (const Line& line : lines_) {
    absl::StrAppend(&code, string(line.indent * spaces_per_indent_, ' '),
                    line.text, "\n");
  }
  return code;
}

void SourceCode::AddLineWithIndent(const string& line) {
  ValidateAndAddLine(current_indent_, line);
}

void SourceCode::AddLineWithoutIndent(const string& line) {
  ValidateAndAddLine(0, line);
}

void SourceCode::AddBlankLine() { ValidateAndAddLine(0, ""); }

void SourceCode::IncreaseIndent() { current_indent_++; }

void SourceCode::DecreaseIndent() { current_indent_--; }

void SourceCode::ValidateAndAddLine(int indent, const string& raw_line) {
  absl::string_view line(raw_line);
  bool had_trailing_newline = absl::ConsumeSuffix(&line, "\n");

  if (absl::StrContains(line, '\n')) {
    LOG(ERROR) << "Attempt to add multiple lines; bad formatting is likely.";
  } else if (had_trailing_newline) {
    LOG(WARNING) << "Superfluous trailing newline in '" << line << "'";
  }
  lines_.push_back({indent, string(absl::StripTrailingAsciiWhitespace(line))});
}

}  // namespace generator
}  // namespace machina
