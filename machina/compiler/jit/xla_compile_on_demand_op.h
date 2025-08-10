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

// The XlaCompileOnDemandOp is an OpKernel that, when its Compute method is
// called, will generate an xla::Computation and run it asynchronously.

#ifndef MACHINA_COMPILER_JIT_MACHINA_MACHINA_XLA_COMPILE_ON_DEMAND_OP_H_
#define MACHINA_COMPILER_JIT_MACHINA_MACHINA_XLA_COMPILE_ON_DEMAND_OP_H_

#include <vector>

#include "machina/compiler/jit/device_compilation_cluster_signature.h"
#include "machina/compiler/jit/device_compilation_profiler.h"
#include "machina/compiler/jit/variable_info.h"
#include "machina/compiler/jit/variable_info_util.h"
#include "machina/compiler/jit/xla_launch_util.h"
#include "machina/compiler/jit/xla_platform_info.h"
#include "machina/compiler/tf2xla/xla_compiler.h"
#include "machina/xla/client/local_client.h"
#include "machina/xla/pjrt/pjrt_client.h"
#include "machina/core/framework/function.h"
#include "machina/core/framework/op_kernel.h"
#include "machina/core/framework/tensor.h"
#include "machina/core/framework/types.h"
#include "machina/core/lib/core/status.h"

namespace machina {

// An OpKernel that compiles an op to an XLA computation and runs it. Unlike
// XlaLaunch this doesn't rely on any rewrites of the graphdef - it will run a
// vanilla TensorFlow op as long as the bridge supports it.
class XlaCompileOnDemandOp : public OpKernel {
 public:
  explicit XlaCompileOnDemandOp(OpKernelConstruction* ctx);
  void Compute(OpKernelContext* ctx) override;

 private:
  absl::Status Compile(const std::vector<XlaCompiler::Argument>& args,
                       OpKernelContext* ctx,
                       DeviceCompiler<xla::LocalExecutable, xla::LocalClient>**
                           xla_device_compiler,
                       DeviceCompilationProfiler** profiler,
                       const XlaCompiler::CompilationResult** result,
                       xla::LocalExecutable** executable);

  absl::Status Compile(const std::vector<XlaCompiler::Argument>& args,
                       OpKernelContext* ctx,
                       DeviceCompiler<xla::PjRtLoadedExecutable,
                                      xla::PjRtClient>** pjrt_device_compiler,
                       DeviceCompilationProfiler** profiler,
                       const XlaCompiler::CompilationResult** result,
                       xla::PjRtLoadedExecutable** executable);

  absl::Status Run(const ResourceVarsSnapshot& variable_args,
                   const XlaCompiler::CompilationResult* result,
                   const DeviceCompiler<xla::LocalExecutable, xla::LocalClient>*
                       xla_device_compiler,
                   xla::LocalExecutable* executable, OpKernelContext* ctx);

  const XlaPlatformInfo platform_info_;

  // Canonicalized function to compile derived from the Op attributes.
  NameAttrList function_;
  DeviceCompilationCanonicalFunction canonical_function_;
};

}  // namespace machina

#endif  // MACHINA_COMPILER_JIT_MACHINA_MACHINA_XLA_COMPILE_ON_DEMAND_OP_H_
