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

#include "machina/compiler/tf2tensorrt/utils/py_utils.h"

#include <string>

#if GOOGLE_CUDA && GOOGLE_TENSORRT
#include "machina/compiler/tf2tensorrt/common/utils.h"
#include "machina/compiler/tf2tensorrt/convert/op_converter_registry.h"
#include "tsl/platform/dso_loader.h"
#include "third_party/tensorrt/NvInfer.h"
#endif

namespace machina {
namespace tensorrt {

bool IsGoogleTensorRTEnabled() {
#if GOOGLE_CUDA && GOOGLE_TENSORRT
#if TF_USE_TENSORRT_STATIC
  LOG(INFO) << "TensorRT libraries are statically linked, skip dlopen check";
  return true;
#else   // TF_USE_TENSORRT_STATIC
  auto handle_or = tsl::internal::DsoLoader::TryDlopenTensorRTLibraries();
  if (!handle_or.ok()) {
    LOG_WARNING_WITH_PREFIX << "Could not find TensorRT";
  }
  return handle_or.ok();
#endif  // TF_USE_TENSORRT_STATIC
#else   // GOOGLE_CUDA && GOOGLE_TENSORRT
  return false;
#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
}

std::vector<std::string> GetRegisteredOpConverters() {
#if GOOGLE_CUDA && GOOGLE_TENSORRT
  auto* registry = machina::tensorrt::convert::GetOpConverterRegistry();
  return registry->ListRegisteredOps();
#else
  return {"undef"};
#endif
}

}  // namespace tensorrt
}  // namespace machina
