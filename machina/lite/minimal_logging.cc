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

#include "machina/lite/minimal_logging.h"

#include <cstdarg>

#include "machina/lite/logger.h"

namespace tflite {
namespace logging_internal {

void MinimalLogger::Log(LogSeverity severity, const char* format, ...) {
  va_list args;
  va_start(args, format);
  LogFormatted(severity, format, args);
  va_end(args);
}

const char* MinimalLogger::GetSeverityName(LogSeverity severity) {
  switch (severity) {
    case TFLITE_LOG_VERBOSE:
      return "VERBOSE";
    case TFLITE_LOG_INFO:
      return "INFO";
    case TFLITE_LOG_WARNING:
      return "WARNING";
    case TFLITE_LOG_ERROR:
      return "ERROR";
    case TFLITE_LOG_SILENT:
      return "SILENT";
  }
  return "<Unknown severity>";
}

LogSeverity MinimalLogger::GetMinimumLogSeverity() {
  return MinimalLogger::minimum_log_severity_;
}

LogSeverity MinimalLogger::SetMinimumLogSeverity(LogSeverity new_severity) {
  LogSeverity old_severity = MinimalLogger::minimum_log_severity_;
  MinimalLogger::minimum_log_severity_ = new_severity;
  return old_severity;
}

}  // namespace logging_internal
}  // namespace tflite
