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

#ifndef MACHINA_COMPILER_TF2TENSORRT_UTILS_TRT_LOGGER_H_
#define MACHINA_COMPILER_TF2TENSORRT_UTILS_TRT_LOGGER_H_

#include "machina/core/platform/types.h"

#if GOOGLE_CUDA && GOOGLE_TENSORRT
#include "third_party/tensorrt/NvInfer.h"

namespace machina {
namespace tensorrt {

// Logger for GIE info/warning/errors
class Logger : public nvinfer1::ILogger {
 public:
  Logger(string name = "DefaultLogger") : name_(name) {}
  void log(nvinfer1::ILogger::Severity severity,
           const char* msg) noexcept override;
  void suppressLoggerMsgs(nvinfer1::ILogger::Severity severity);
  void unsuppressLoggerMsgs(nvinfer1::ILogger::Severity severity);
  void unsuppressAllLoggerMsgs() { suppressedMsg_ = 0; }
  static Logger* GetLogger();

 private:
  bool isValidSeverity(nvinfer1::ILogger::Severity severity,
                       const char* msg = nullptr) noexcept;
  const string name_;
  unsigned int suppressedMsg_ = 0;
};

}  // namespace tensorrt
}  // namespace machina

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT

#endif  // MACHINA_COMPILER_TF2TENSORRT_UTILS_TRT_LOGGER_H_
