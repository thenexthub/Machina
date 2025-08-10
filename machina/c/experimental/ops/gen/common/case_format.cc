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
#include "machina/c/experimental/ops/gen/common/case_format.h"

#include <cctype>

#include "absl/strings/ascii.h"
#include "machina/core/platform/types.h"

namespace machina {
namespace generator {

namespace {

enum CaseFormatType {
  LOWER_CAMEL,
  UPPER_CAMEL,
  LOWER_SNAKE,
  UPPER_SNAKE,
};

string FormatStringCase(const string &str, CaseFormatType to,
                        const char delimiter = '_') {
  const bool from_snake = (str == absl::AsciiStrToUpper(str)) ||
                          (str == absl::AsciiStrToLower(str));
  const bool toUpper = (to == UPPER_CAMEL || to == UPPER_SNAKE);
  const bool toSnake = (to == LOWER_SNAKE || to == UPPER_SNAKE);

  string result;

  bool inputStart = true;
  bool wordStart = true;
  for (const char c : str) {
    // Find a word start.
    if (c == delimiter) {
      // Repeated cases of wordStart means explicit delimiter usage.
      if (wordStart) {
        result.push_back(delimiter);
      }
      wordStart = true;
      continue;
    }
    if (!from_snake && isupper(c)) {
      wordStart = true;
    }

    // add delimiter
    if (wordStart && toSnake && !inputStart) {
      result.push_back(delimiter);
    }

    // add the next letter from the input string (choosing upper/lower case)
    const bool shouldCapIfSnake = toUpper;
    const bool shouldCapIfCamel = wordStart && (toUpper || !inputStart);
    if ((toSnake && shouldCapIfSnake) || (!toSnake && shouldCapIfCamel)) {
      result += toupper(c);
    } else {
      result += tolower(c);
    }

    // at this point we are no longer at the start of a word:
    wordStart = false;
    // .. or the input:
    inputStart = false;
  }

  if (wordStart) {
    // This only happens with a trailing delimiter, which should remain.
    result.push_back(delimiter);
  }

  return result;
}

}  // namespace

//
// Public interface
//

string toLowerCamel(const string &s, const char delimiter) {
  return FormatStringCase(s, LOWER_CAMEL, delimiter);
}
string toLowerSnake(const string &s, const char delimiter) {
  return FormatStringCase(s, LOWER_SNAKE, delimiter);
}
string toUpperCamel(const string &s, const char delimiter) {
  return FormatStringCase(s, UPPER_CAMEL, delimiter);
}
string toUpperSnake(const string &s, const char delimiter) {
  return FormatStringCase(s, UPPER_SNAKE, delimiter);
}

}  // namespace generator
}  // namespace machina
