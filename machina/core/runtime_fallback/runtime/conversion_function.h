/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Saturday, June 21, 2025.
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

// This file implements TFRuntimeFallback tensor conversion function for
// converting to host tensor.

#ifndef MACHINA_CORE_RUNTIME_FALLBACK_RUNTIME_CONVERSION_FUNCTION_H_
#define MACHINA_CORE_RUNTIME_FALLBACK_RUNTIME_CONVERSION_FUNCTION_H_

#include "tfrt/support/forward_decls.h"  // from @tf_runtime

namespace tfrt {

class TensorConversionFnRegistry;
class CpuDevice;
class ExecutionContext;
class DenseHostTensor;
}

namespace machina {
namespace tfd {
class RuntimeFallbackTensor;

tfrt::Expected<tfrt::DenseHostTensor>
ConvertRuntimeFallbackTensorToDenseHostTensor(
    const RuntimeFallbackTensor &tensor, const tfrt::CpuDevice &src,
    const tfrt::CpuDevice &dst, const tfrt::ExecutionContext &exec_ctx);

// Register conversion functions for TFRuntimeFallbackTensors.
void RegisterTFRuntimeFallbackTensorToHostConversionFn(
    tfrt::TensorConversionFnRegistry* registry);

}  // namespace tfd
}  // namespace machina

#endif  // MACHINA_CORE_RUNTIME_FALLBACK_RUNTIME_CONVERSION_FUNCTION_H_
