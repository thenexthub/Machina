/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, August 10, 2025.
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

#ifndef MACHINA_COMPILER_TF2MACHINA_MACHINA_XLA_MACHINA_MACHINA_XLA_COMPILED_CPU_FUNCTION_THUNKS_H_
#define MACHINA_COMPILER_TF2MACHINA_MACHINA_XLA_MACHINA_MACHINA_XLA_COMPILED_CPU_FUNCTION_THUNKS_H_

#include <cassert>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "machina/compiler/tf2xla/xla_compiled_cpu_function.h"
#include "machina/xla/backends/cpu/nanort/nanort_executable.h"
#include "machina/xla/executable_run_options.h"
#include "machina/xla/service/cpu/executable.pb.h"
#include "machina/xla/tsl/platform/threadpool.h"

namespace machina {

class XlaCompiledCpuFunctionThunks : public XlaCompiledCpuFunction {
 public:
  explicit XlaCompiledCpuFunctionThunks(
      const StaticData& static_data,
      AllocMode alloc_mode =
          AllocMode::ARGS_VARIABLES_RESULTS_PROFILES_AND_TEMPS);

  bool Run() override;

  bool is_thunk_mode() const override { return true; }

  void set_thread_pool(const Eigen::ThreadPoolDevice* pool) override {
    thunk_run_options_.set_intra_op_thread_pool(pool);
  }

 protected:
  std::vector<xla::cpu::NanoRtExecutable::Argument> GenerateNanortArgs();
  std::vector<xla::cpu::NanoRtExecutable::Result> GenerateNanortResults();
  xla::cpu::NanoRtExecutable::PreallocatedTemp GenerateNanortPreallocatedTemp();

 private:
  // For NanoRtExecutable.
  absl::Status Execute(
      absl::Span<const xla::cpu::NanoRtExecutable::Argument> arguments,
      absl::Span<const xla::cpu::NanoRtExecutable::Result> results,
      xla::cpu::NanoRtExecutable::PreallocatedTemp temp);

  std::unique_ptr<xla::cpu::NanoRtExecutable> executable_;
  xla::cpu::NanoRtExecutable::ExecuteOptions thunk_run_options_;
};

}  // namespace machina

#endif  // MACHINA_COMPILER_TF2MACHINA_MACHINA_XLA_MACHINA_MACHINA_XLA_COMPILED_CPU_FUNCTION_THUNKS_H_
