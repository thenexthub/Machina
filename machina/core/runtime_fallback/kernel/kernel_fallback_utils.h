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
#ifndef MACHINA_CORE_RUNTIME_FALLBACK_KERNEL_KERNEL_FALLBACK_UTILS_H_
#define MACHINA_CORE_RUNTIME_FALLBACK_KERNEL_KERNEL_FALLBACK_UTILS_H_

#include <functional>

#include "machina/core/framework/device.h"
#include "machina/core/runtime_fallback/kernel/kernel_fallback_compat_request_state.h"
#include "machina/core/tfrt/fallback/op_kernel_runner.h"
#include "machina/core/tfrt/utils/fallback_tensor.h"
#include "tfrt/host_context/async_value.h"  // from @tf_runtime
#include "tfrt/host_context/sync_kernel_utils.h"  // from @tf_runtime
#include "tfrt/host_context/value.h"  // from @tf_runtime
#include "tfrt/support/forward_decls.h"  // from @tf_runtime
#include "tfrt/support/variant.h"  // from @tf_runtime

namespace machina {
namespace tfd {

std::function<void(std::function<void()>)>* GetDefaultRunner();

using TfInputs =
    tfrt::Variant<tfrt::Monostate, toolchain::ArrayRef<tfrt::AsyncValue*>,
                  tfrt::RepeatedSyncArguments<tfrt_stub::FallbackTensor>&>;

// Sets up the OpKernelcontext::Params in `run_state` with the objects and data
// in `runner`, `fallback_request_state` and `device`.
void SetUpParams(const machina::tfrt_stub::OpKernelRunner& runner,
                 const KernelFallbackCompatRequestState& fallback_request_state,
                 machina::Device* device,
                 machina::tfrt_stub::OpKernelRunState& run_state);

// Return the device to be used for the fallback kernel execution. The device is
// guaranteed to be alive during the graph execution.
machina::Device* GetDeviceFromFallbackState(
    const KernelFallbackCompatRequestState& fallback_request_state,
    const tfrt_stub::OpKernelRunner& kernel_runner);

}  // namespace tfd
}  // namespace machina

#endif  // MACHINA_CORE_RUNTIME_FALLBACK_KERNEL_KERNEL_FALLBACK_UTILS_H_
