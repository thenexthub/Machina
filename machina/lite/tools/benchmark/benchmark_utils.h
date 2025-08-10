/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, May 15, 2025.
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

#ifndef MACHINA_LITE_TOOLS_BENCHMARK_BENCHMARK_UTILS_H_
#define MACHINA_LITE_TOOLS_BENCHMARK_BENCHMARK_UTILS_H_

#include <sstream>
#include <string>
#include <vector>

namespace tflite {
namespace benchmark {
namespace util {

// A convenient function that wraps tflite::profiling::time::SleepForMicros and
// simply return if 'sleep_seconds' is negative.
void SleepForSeconds(double sleep_seconds);

// Split the 'str' according to 'delim', and store each splitted element into
// 'values'.
template <typename T>
bool SplitAndParse(const std::string& str, char delim, std::vector<T>* values) {
  std::istringstream input(str);
  for (std::string line; std::getline(input, line, delim);) {
    std::istringstream to_parse(line);
    T val;
    to_parse >> val;
    if (!to_parse.eof() && !to_parse.good()) {
      return false;
    }
    values->emplace_back(val);
  }
  return true;
}

}  // namespace util
}  // namespace benchmark
}  // namespace tflite

#endif  // MACHINA_LITE_TOOLS_BENCHMARK_BENCHMARK_UTILS_H_
