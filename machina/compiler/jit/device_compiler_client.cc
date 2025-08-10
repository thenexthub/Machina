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

#include "machina/compiler/jit/device_compiler_client.h"

#include "machina/compiler/tf2xla/xla_compiler.h"
#include "machina/core/util/determinism.h"

namespace machina {

xla::ExecutableBuildOptions GetExecutableBuildOptions(
    const XlaCompiler::Options& options,
    const XlaCompiler::CompilationResult& result, int default_device_ordinal) {
  xla::ExecutableBuildOptions build_options;
  if (result.collective_info) {
    build_options.set_num_replicas(result.collective_info->group_size);
  }
  if (options.device_ordinal != -1) {
    build_options.set_device_ordinal(options.device_ordinal);
  } else if (default_device_ordinal != -1) {
    build_options.set_device_ordinal(default_device_ordinal);
  }
  build_options.set_result_layout(result.xla_output_shape);
  build_options.set_device_allocator(options.device_allocator.get());
  build_options.set_alias_passthrough_params(options.alias_passthrough_params);
  build_options.mutable_debug_options()->set_xla_detailed_logging(
      options.detailed_logging);
  if (machina::OpDeterminismRequired()) {
    build_options.mutable_debug_options()->set_xla_gpu_deterministic_ops(true);
  }
  return build_options;
}

}  // namespace machina
