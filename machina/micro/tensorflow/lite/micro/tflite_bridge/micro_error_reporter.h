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
#ifndef MACHINA_LITE_MICRO_TFLITE_BRIDGE_MICRO_ERROR_REPORTER_H_
#define MACHINA_LITE_MICRO_TFLITE_BRIDGE_MICRO_ERROR_REPORTER_H_

#include <cstdarg>

#include "machina/lite/core/api/error_reporter.h"
#include "machina/lite/micro/compatibility.h"

namespace tflite {
// Get a pointer to a singleton global error reporter.
ErrorReporter* GetMicroErrorReporter();
class MicroErrorReporter : public ErrorReporter {
 public:
  ~MicroErrorReporter() override {}
  int Report(const char* format, va_list args) override;

 private:
  TF_LITE_REMOVE_VIRTUAL_DELETE
};

}  // namespace tflite

#endif  // MACHINA_LITE_MICRO_TFLITE_BRIDGE_MICRO_ERROR_REPORTER_H_
