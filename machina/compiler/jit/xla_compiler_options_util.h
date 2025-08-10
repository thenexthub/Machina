/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, July 12, 2025.
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

#ifndef MACHINA_COMPILER_JIT_MACHINA_MACHINA_XLA_COMPILER_OPTIONS_UTIL_H_
#define MACHINA_COMPILER_JIT_MACHINA_MACHINA_XLA_COMPILER_OPTIONS_UTIL_H_

#include "machina/compiler/jit/device_compiler.h"
#include "machina/compiler/jit/xla_platform_info.h"
#include "machina/compiler/tf2xla/xla_compiler.h"
#include "machina/xla/client/local_client.h"
#include "machina/xla/pjrt/pjrt_client.h"
#include "machina/xla/stream_executor/stream.h"
#include "machina/core/framework/device_base.h"
#include "machina/core/framework/function.h"

namespace machina {

// Returns created options for the XLA compiler.
XlaCompiler::Options GenerateCompilerOptions(
    const DeviceCompiler<xla::LocalExecutable, xla::LocalClient>&
        xla_device_compiler,
    const FunctionLibraryRuntime& function_library, DeviceBase* device,
    se::Stream* stream, const XlaPlatformInfo& platform_info,
    bool has_ref_vars);

// Returns created options for XLA compiler when TFRT-TPU is used.
XlaCompiler::Options GenerateCompilerOptionsForTfrtTpu(
    const DeviceCompiler<xla::LocalExecutable, xla::LocalClient>&
        xla_device_compiler,
    const FunctionLibraryRuntime& function_library);

// Returns created options for XLA compiler when PjRt (Device API) is used for
// compilation and execution.
XlaCompiler::Options GenerateCompilerOptionsForPjRt(
    const FunctionLibraryRuntime& function_library,
    const DeviceBase* device_base, const XlaPlatformInfo& platform_info,
    const DeviceCompiler<xla::PjRtLoadedExecutable, xla::PjRtClient>*
        pjrt_device_compiler);

// Returns created options for XLA compiler when PjRt (Device API) is used for
// compilation and execution.
XlaCompiler::Options GenerateCompilerOptionsForPjRt(
    const FunctionLibraryDefinition* function_library_def,
    int graph_def_version, const DeviceBase* device_base,
    const XlaPlatformInfo& platform_info,
    const DeviceCompiler<xla::PjRtLoadedExecutable, xla::PjRtClient>*
        pjrt_device_compiler);

// Returns created CompileOptions for XLA compiler.
XlaCompiler::CompileOptions GenerateCompileOptions(
    bool has_ref_vars, bool may_alias_resource_update);

}  // namespace machina

#endif  // MACHINA_COMPILER_JIT_MACHINA_MACHINA_XLA_COMPILER_OPTIONS_UTIL_H_
