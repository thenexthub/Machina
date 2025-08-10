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
#ifndef MACHINA_COMPILER_TF2TENSORRT_CONVERT_LOGGER_REGISTRY_H_
#define MACHINA_COMPILER_TF2TENSORRT_CONVERT_LOGGER_REGISTRY_H_

#include "machina/core/lib/core/status.h"
#include "machina/core/platform/macros.h"
#include "machina/core/platform/types.h"

#if GOOGLE_CUDA && GOOGLE_TENSORRT

#include "third_party/tensorrt/NvInfer.h"

namespace machina {
namespace tensorrt {

class LoggerRegistry {
 public:
  virtual Status Register(const string& name, nvinfer1::ILogger* logger) = 0;
  virtual nvinfer1::ILogger* LookUp(const string& name) = 0;
  virtual ~LoggerRegistry() {}
};

LoggerRegistry* GetLoggerRegistry();

class RegisterLogger {
 public:
  RegisterLogger(const string& name, nvinfer1::ILogger* logger) {
    TF_CHECK_OK(GetLoggerRegistry()->Register(name, logger));
  }
};

#define REGISTER_TENSORRT_LOGGER(name, logger) \
  REGISTER_TENSORRT_LOGGER_UNIQ_HELPER(__COUNTER__, name, logger)
#define REGISTER_TENSORRT_LOGGER_UNIQ_HELPER(ctr, name, logger) \
  REGISTER_TENSORRT_LOGGER_UNIQ(ctr, name, logger)
#define REGISTER_TENSORRT_LOGGER_UNIQ(ctr, name, logger)                 \
  static ::machina::tensorrt::RegisterLogger register_trt_logger##ctr \
      TF_ATTRIBUTE_UNUSED =                                              \
          ::machina::tensorrt::RegisterLogger(name, logger)

}  // namespace tensorrt
}  // namespace machina

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
#endif  // MACHINA_COMPILER_TF2TENSORRT_CONVERT_LOGGER_REGISTRY_H_
