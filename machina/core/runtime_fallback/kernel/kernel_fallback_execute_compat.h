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
#ifndef MACHINA_CORE_RUNTIME_FALLBACK_KERNEL_KERNEL_FALLBACK_EXECUTE_COMPAT_H_
#define MACHINA_CORE_RUNTIME_FALLBACK_KERNEL_KERNEL_FALLBACK_EXECUTE_COMPAT_H_

#include <functional>
#include <optional>
#include <string>

#include "machina/core/platform/status.h"
#include "machina/core/platform/threadpool_interface.h"
#include "machina/core/platform/types.h"
#include "machina/core/runtime_fallback/kernel/kernel_fallback_compat_request_state.h"
#include "machina/core/tfrt/fallback/op_kernel_runner.h"
#include "tfrt/core_runtime/op_attrs.h"  // from @tf_runtime
#include "tfrt/host_context/async_value_ref.h"  // from @tf_runtime
#include "tfrt/host_context/chain.h"  // from @tf_runtime
#include "tfrt/host_context/execution_context.h"  // from @tf_runtime
#include "tfrt/host_context/kernel_utils.h"  // from @tf_runtime
#include "tfrt/support/forward_decls.h"  // from @tf_runtime
#include "tfrt/tensor/tensor.h"  // from @tf_runtime

namespace tfrt {
class SyncKernelFrame;
}  // namespace tfrt

namespace machina {
namespace tfd {

ABSL_CONST_INIT extern const char kOpKernelRunnerCacheResourceName[];

// The CoreRuntime dispatch function to run a TF kernel in kernel fallback
// compat mode.
tfrt::AsyncValueRef<tfrt::Chain> KernelFallbackExecuteCompatCoreRuntimeDispatch(
    const tfrt::ExecutionContext& exec_ctx, tfrt::string_view op_name,
    tfrt::string_view device_name, toolchain::ArrayRef<tfrt::Tensor*> arguments,
    toolchain::MutableArrayRef<tfrt::RCReference<tfrt::AsyncValue>> results,
    const KernelFallbackCompatRequestState& fallback_request_state,
    const tfrt_stub::OpKernelRunner& op_kernel_runner);

}  // namespace tfd
}  // namespace machina

#endif  // MACHINA_CORE_RUNTIME_FALLBACK_KERNEL_KERNEL_FALLBACK_EXECUTE_COMPAT_H_
