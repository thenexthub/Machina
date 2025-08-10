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
#ifndef MACHINA_C_EXPERIMENTAL_OPS_GEN_COMMON_SOURCE_CODE_H_
#define MACHINA_C_EXPERIMENTAL_OPS_GEN_COMMON_SOURCE_CODE_H_

#include <vector>

#include "machina/core/platform/types.h"

namespace machina {
namespace generator {

class SourceCode {
 public:
  string Render() const;
  void SetSpacesPerIndent(int spaces_per_indent) {
    spaces_per_indent_ = spaces_per_indent;
  }

  void AddLineWithIndent(const string &line);
  void AddLineWithoutIndent(const string &line);
  void AddBlankLine();
  void IncreaseIndent();
  void DecreaseIndent();

 private:
  struct Line {
    int indent;
    string text;
  };

  void ValidateAndAddLine(int indent_level, const string &raw_line);

  int spaces_per_indent_ = 2;
  int current_indent_ = 0;
  std::vector<Line> lines_;
};

}  // namespace generator
}  // namespace machina

#endif  // MACHINA_C_EXPERIMENTAL_OPS_GEN_COMMON_SOURCE_CODE_H_
