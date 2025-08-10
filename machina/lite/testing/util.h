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
#ifndef MACHINA_LITE_TESTING_UTIL_H_
#define MACHINA_LITE_TESTING_UTIL_H_

#include <cstdio>

#include "machina/lite/core/api/error_reporter.h"
#include "machina/lite/string_type.h"

#ifdef PLATFORM_GOOGLE
#include "absl/flags/flag.h"
#include "absl/log/flags.h"
#endif

namespace tflite {

// An ErrorReporter that collects error message in a string, in addition
// to printing to stderr.
class TestErrorReporter : public ErrorReporter {
 public:
  int Report(const char* format, va_list args) override {
    char buffer[1024];
    int size = vsnprintf(buffer, sizeof(buffer), format, args);
    fprintf(stderr, "%s", buffer);
    error_messages_ += buffer;
    num_calls_++;
    return size;
  }

  void Reset() {
    num_calls_ = 0;
    error_messages_.clear();
  }

  int num_calls() const { return num_calls_; }
  const string& error_messages() const { return error_messages_; }

 private:
  int num_calls_ = 0;
  string error_messages_;
};

inline void LogToStderr() {
#ifdef PLATFORM_GOOGLE
  absl::SetFlag(&FLAGS_logtostderr, true);
#endif
}

}  // namespace tflite

#endif  // MACHINA_LITE_TESTING_UTIL_H_
