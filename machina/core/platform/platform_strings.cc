/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 29, 2025.
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

#include "machina/core/platform/platform_strings.h"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace machina {

int GetPlatformStrings(const std::string& path,
                       std::vector<std::string>* found) {
  int result;
  FILE* ifp = fopen(path.c_str(), "rb");
  if (ifp != nullptr) {
    static const char prefix[] = TF_PLAT_STR_MAGIC_PREFIX_;
    int first_char = prefix[1];
    int last_char = -1;
    int c;
    while ((c = getc(ifp)) != EOF) {
      if (c == first_char && last_char == 0) {
        int i = 2;
        while (prefix[i] != 0 && (c = getc(ifp)) == prefix[i]) {
          i++;
        }
        if (prefix[i] == 0) {
          std::string str;
          while ((c = getc(ifp)) != EOF && c != 0) {
            str.push_back(c);
          }
          if (!str.empty()) {
            found->push_back(str);
          }
        }
      }
      last_char = c;
    }

    result = (ferror(ifp) == 0) ? 0 : errno;
    if (fclose(ifp) != 0) {
      result = errno;
    }
  } else {
    result = errno;
  }
  return result;
}

}  // namespace machina
