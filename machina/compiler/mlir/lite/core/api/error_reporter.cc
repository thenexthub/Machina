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
#include "machina/compiler/mlir/lite/core/api/error_reporter.h"

#include <cstdarg>

namespace tflite {

int ErrorReporter::Report(const char* format, ...) {
  va_list args;
  va_start(args, format);
  int code = Report(format, args);
  va_end(args);
  return code;
}

// TODO(aselle): Make the name of ReportError on context the same, so
// we can use the ensure functions w/o a context and w/ a reporter.
int ErrorReporter::ReportError(void*, const char* format, ...) {
  va_list args;
  va_start(args, format);
  int code = Report(format, args);
  va_end(args);
  return code;
}

}  // namespace tflite
