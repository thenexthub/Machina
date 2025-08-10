/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 8, 2025.
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
#ifndef MACHINA_LITE_TESTING_JOIN_H_
#define MACHINA_LITE_TESTING_JOIN_H_

#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <sstream>

#include "machina/lite/string_type.h"

namespace tflite {
namespace testing {

// Join a list of data with default precision separated by delimiter.
template <typename T>
string JoinDefault(T* data, size_t len, const string& delimiter) {
  if (len == 0 || data == nullptr) {
    return "";
  }
  std::stringstream result;
  result << data[0];
  for (int i = 1; i < len; i++) {
    result << delimiter << data[i];
  }
  return result.str();
}

// Join a list of data with fixed precision separated by delimiter.
template <typename T>
string Join(T* data, size_t len, const string& delimiter) {
  if (len == 0 || data == nullptr) {
    return "";
  }
  std::stringstream result;
  result << std::setprecision(9) << data[0];
  for (int i = 1; i < len; i++) {
    result << std::setprecision(9) << delimiter << data[i];
  }
  return result.str();
}

// Join a list of uint8 data separated by a delimiter. Cast data to int before
// placing it in the string to prevent values from being treated like chars.
template <>
inline string Join<uint8_t>(uint8_t* data, size_t len,
                            const string& delimiter) {
  if (len == 0 || data == nullptr) {
    return "";
  }
  std::stringstream result;
  result << static_cast<int>(data[0]);
  for (int i = 1; i < len; i++) {
    result << delimiter << static_cast<int>(data[i]);
  }
  return result.str();
}

// Join a list of int8 data separated by a delimiter. Cast data to int before
// placing it in the string to prevent values from being treated like chars.
template <>
inline string Join<int8_t>(int8_t* data, size_t len, const string& delimiter) {
  if (len == 0 || data == nullptr) {
    return "";
  }
  std::stringstream result;
  result << static_cast<int>(data[0]);
  for (int i = 1; i < len; i++) {
    result << delimiter << static_cast<int>(data[i]);
  }
  return result.str();
}

}  // namespace testing
}  // namespace tflite

#endif  // MACHINA_LITE_TESTING_JOIN_H_
