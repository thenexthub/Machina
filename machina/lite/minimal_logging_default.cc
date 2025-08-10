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

#include <stdarg.h>

#include <cstdio>

#include "machina/lite/minimal_logging.h"

namespace tflite {
namespace logging_internal {

#ifndef NDEBUG
// In debug builds, default is VERBOSE.
LogSeverity MinimalLogger::minimum_log_severity_ = TFLITE_LOG_VERBOSE;
#else
// In prod builds, default is INFO.
LogSeverity MinimalLogger::minimum_log_severity_ = TFLITE_LOG_INFO;
#endif

void MinimalLogger::LogFormatted(LogSeverity severity, const char* format,
                                 va_list args) {
  if (severity >= MinimalLogger::minimum_log_severity_) {
    fprintf(stderr, "%s: ", GetSeverityName(severity));
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wformat-nonliteral"
    vfprintf(stderr, format, args);
#pragma clang diagnostic pop
    fputc('\n', stderr);
  }
}

}  // namespace logging_internal
}  // namespace tflite
