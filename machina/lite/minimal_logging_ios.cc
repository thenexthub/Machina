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

#include <syslog.h>

#include <cstdarg>
#include <cstdio>

#include "machina/lite/minimal_logging.h"

namespace tflite {
namespace logging_internal {
namespace {

int GetPlatformSeverity(LogSeverity severity) {
  switch (severity) {
    case TFLITE_LOG_VERBOSE:
    case TFLITE_LOG_INFO:
      return LOG_INFO;
    case TFLITE_LOG_WARNING:
      return LOG_WARNING;
    case TFLITE_LOG_ERROR:
      return LOG_ERR;
    // LOG_SILENT doesn't exist, Use LOG_DEBUG instead.
    case TFLITE_LOG_SILENT:
    default:
      return LOG_DEBUG;
  }
}

}  // namespace

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
    // First log to iOS system logging API.
    va_list args_copy;
    va_copy(args_copy, args);
    // TODO(b/123704468): Use os_log when available.
    vsyslog(GetPlatformSeverity(severity), format, args_copy);
    va_end(args_copy);

    // Also print to stderr for standard console applications.
    fprintf(stderr, "%s: ", GetSeverityName(severity));
    va_copy(args_copy, args);
    vfprintf(stderr, format, args_copy);
    va_end(args_copy);
    fputc('\n', stderr);
  }
}

}  // namespace logging_internal
}  // namespace tflite
