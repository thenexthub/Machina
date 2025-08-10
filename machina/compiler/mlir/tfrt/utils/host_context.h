/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, June 15, 2025.
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

#ifndef MACHINA_COMPILER_MLIR_TFRT_UTILS_HOST_CONTEXT_H_
#define MACHINA_COMPILER_MLIR_TFRT_UTILS_HOST_CONTEXT_H_

#include <cstdint>
#include <memory>

#include "absl/base/attributes.h"
#include "tfrt/host_context/host_context.h"  // from @tf_runtime

namespace machina {

// The name of the default host device for running fallback kernels.
ABSL_CONST_INIT extern const char* const kDefaultHostDeviceName;

std::unique_ptr<tfrt::HostContext> CreateSingleThreadedHostContext();
std::unique_ptr<tfrt::HostContext> CreateMultiThreadedHostContext(
    int64_t num_threads);

}  // namespace machina

#endif  // MACHINA_COMPILER_MLIR_TFRT_UTILS_HOST_CONTEXT_H_
