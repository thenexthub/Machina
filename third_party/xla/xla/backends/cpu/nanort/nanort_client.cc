/* Copyright 2024 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "machina/xla/backends/cpu/nanort/nanort_client.h"

#include <memory>
#include <utility>

#include "absl/status/statusor.h"
#include "machina/xla/backends/cpu/nanort/nanort_executable.h"
#include "machina/xla/debug_options_flags.h"
#include "machina/xla/hlo/builder/xla_computation.h"
#include "machina/xla/hlo/ir/hlo_module.h"
#include "machina/xla/pjrt/utils.h"
#include "machina/xla/service/compiler.h"
#include "machina/xla/service/cpu/cpu_compiler.h"
#include "machina/xla/service/dump.h"
#include "machina/xla/service/executable.h"
#include "machina/xla/service/hlo_module_config.h"
#include "machina/xla/shape.h"
#include "machina/xla/tsl/platform/logging.h"
#include "machina/xla/tsl/platform/statusor.h"
#include "machina/xla/util.h"
#include "tsl/profiler/lib/traceme.h"
#include "tsl/profiler/lib/traceme_encode.h"

namespace xla::cpu {

using ::tsl::profiler::TraceMe;
using ::tsl::profiler::TraceMeEncode;

absl::StatusOr<std::unique_ptr<NanoRtExecutable>> NanoRtClient::Compile(
    const XlaComputation& computation) {
  TraceMe trace([&] {
    return TraceMeEncode("NanoRtClient::Compile",
                         {{"computation", computation.name()}});
  });

  TF_ASSIGN_OR_RETURN(ProgramShape program_shape,
                      computation.GetProgramShape());

  HloModuleConfig hlo_module_config(program_shape, /*ignore_layouts=*/false);
  hlo_module_config.set_debug_options(GetDebugOptionsFromFlags());

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloModule> hlo_module,
      HloModule::CreateFromProto(computation.proto(), hlo_module_config));

  static constexpr char kBeforeOptimizationsDumpName[] = "before_optimizations";
  DumpHloModuleIfEnabled(*hlo_module, kBeforeOptimizationsDumpName);

  // Use default XLA compiler options.
  Compiler::CompileOptions compile_options;

  // Run high-level XLA CPU compiler passes.
  cpu::CpuCompiler compiler;
  TF_ASSIGN_OR_RETURN(hlo_module, compiler.RunHloPasses(std::move(hlo_module),
                                                        /*stream_exec=*/nullptr,
                                                        compile_options));

  auto optimized_hlo_program_shape =
      hlo_module->entry_computation_layout().ComputeProgramShape();

  // Compile optimized HLO module to CPU executable.
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<Executable> executable,
      compiler.RunBackend(std::move(hlo_module), /*stream_exec=*/nullptr,
                          compile_options));

  return NanoRtExecutable::Create(std::move(executable),
                                  optimized_hlo_program_shape);
}

}  // namespace xla::cpu
