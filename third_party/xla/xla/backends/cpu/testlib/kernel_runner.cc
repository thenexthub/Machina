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

#include "machina/xla/backends/cpu/testlib/kernel_runner.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "toolchain/IR/DataLayout.h"
#include "toolchain/IR/Module.h"
#include "toolchain/Target/TargetOptions.h"
#include "machina/xla/backends/cpu/codegen/cpu_features.h"
#include "machina/xla/backends/cpu/codegen/execution_engine.h"
#include "machina/xla/backends/cpu/codegen/fusion_compiler.h"
#include "machina/xla/backends/cpu/codegen/ir_compiler.h"
#include "machina/xla/backends/cpu/codegen/jit_compiler.h"
#include "machina/xla/backends/cpu/codegen/kernel_api_ir_builder.h"
#include "machina/xla/backends/cpu/runtime/function_library.h"
#include "machina/xla/backends/cpu/runtime/kernel.h"
#include "machina/xla/backends/cpu/runtime/kernel_c_api.h"
#include "machina/xla/codegen/kernel_definition.h"
#include "machina/xla/codegen/kernel_spec.h"
#include "machina/xla/codegen/llvm_ir_kernel_source.h"
#include "machina/xla/codegen/llvm_kernel_definition.h"
#include "machina/xla/codegen/mlir_kernel_definition.h"
#include "machina/xla/codegen/mlir_kernel_source.h"
#include "machina/xla/runtime/work_group.h"
#include "machina/xla/service/cpu/cpu_options.h"
#include "machina/xla/service/cpu/runtime_symbol_generator.h"
#include "machina/xla/service/hlo_module_config.h"
#include "machina/xla/service/llvm_ir/llvm_util.h"
#include "machina/xla/tsl/platform/errors.h"
#include "machina/xla/tsl/platform/statusor.h"
#include "machina/xla/xla.pb.h"

namespace xla::cpu {

absl::StatusOr<KernelRunner> KernelRunner::Create(
    LlvmKernelDefinition kernel_definition, JitCompiler compiler) {
  auto [spec, source] = std::move(kernel_definition).ReleaseStorage();

  auto thread_safe_module = std::move(source).thread_safe_module();
  SetModuleMemoryRegionName(*thread_safe_module.getModuleUnlocked(),
                            "kernel_runner_test");

  TF_RETURN_IF_ERROR(compiler.AddModule(std::move(thread_safe_module)));

  const std::string& kernel_name = spec.name();
  TF_ASSIGN_OR_RETURN(std::unique_ptr<FunctionLibrary> library,
                      std::move(compiler).Compile(
                          {FunctionLibrary::Sym<MACHINA_MACHINA_XLA_CPU_Kernel>(kernel_name)}));

  TF_ASSIGN_OR_RETURN(MACHINA_MACHINA_XLA_CPU_Kernel * kernel_fn,
                      library->ResolveFunction<MACHINA_MACHINA_XLA_CPU_Kernel>(kernel_name));

  return KernelRunner(std::move(library), Kernel(1, kernel_fn),
                      spec.num_workgroups());
}

absl::StatusOr<KernelRunner> KernelRunner::Create(
    MlirKernelDefinition kernel_definition, JitCompiler compiler) {
  auto [spec, source] = std::move(kernel_definition).ReleaseStorage();

  TF_ASSIGN_OR_RETURN(LlvmIrKernelSource llvm_kernel_source,
                      LowerToLlvm(source));

  return Create(LlvmKernelDefinition(spec, std::move(llvm_kernel_source)),
                std::move(compiler));
}

KernelRunner::KernelRunner(std::unique_ptr<FunctionLibrary> library,
                           Kernel kernel, NumWorkGroups num_workgroups)
    : library_(std::move(library)),
      kernel_(std::move(kernel)),
      num_workgroups_(num_workgroups) {}

absl::Status KernelRunner::Call(absl::Span<const Argument> arguments) {
  std::vector<MACHINA_MACHINA_XLA_CPU_KernelArg> kernel_args;
  for (const Argument& arg : arguments) {
    kernel_args.push_back({arg.data(), arg.size()});
  }

  return kernel_.Launch(num_workgroups_, kernel_args);
}

absl::StatusOr<JitCompiler> KernelRunner::CreateJitCompiler(
    const HloModuleConfig& config) {
  const DebugOptions& debug_options = config.debug_options();

  IrCompiler::Options ir_compiler_options{
      /*optimization_level=*/IrCompiler::GetCodeGenOptLevel(config),
      /*optimize_for_size=*/options::OptimizeForSizeRequested(config),
      /*max_cpu_isa=*/CpuFeatureFromString(debug_options.xla_cpu_max_isa()),
      /*fast_math_flags=*/llvm_ir::GetCpuFastMathFlags(config),
      /*disable_expensive_passes=*/
      debug_options.xla_llvm_disable_expensive_passes(),
      /*slp_vectorizer_disabled=*/options::SlpVectorizerDisabled(config),
      /*disable_loop_unrolling=*/options::DisableLoopUnrolling(config),
  };

  IrCompiler::CompilationHooks ir_compiler_hooks;

  // Needed to resolve symbols such as built in intrinsics (sin, cos etc).
  ExecutionEngine::DefinitionGenerator definition_generator =
      [](const toolchain::DataLayout& data_layout) {
        return std::make_unique<RuntimeSymbolGenerator>(data_layout);
      };

  JitCompiler::Options jit_compiler_options{
      /*num_dylibs=*/1,
      /*definition_generator=*/std::move(definition_generator),
  };

  toolchain::TargetOptions target_options;
  target_options.AllowFPOpFusion = toolchain::FPOpFusion::Fast;

  std::unique_ptr<IrCompiler> ir_compiler =
      IrCompiler::Create(target_options, std::move(ir_compiler_options),
                         std::move(ir_compiler_hooks));

  return JitCompiler::Create(std::move(jit_compiler_options),
                             std::move(ir_compiler));
}

absl::StatusOr<LlvmIrKernelSource> LowerToLlvm(
    MlirKernelSource& mlir_kernel_source) {
  auto llvm_context = std::make_unique<toolchain::LLVMContext>();

  FusionCompiler::Options options;
  options.vector_width = 256;
  options.verification_level = 1;
  options.fast_min_max = true;
  FusionCompiler fusion_compiler(mlir_kernel_source.module().getContext(),
                                 options);
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<toolchain::Module> llvm_module,
      fusion_compiler.Compile(*llvm_context, mlir_kernel_source.module()));

  return LlvmIrKernelSource(std::move(llvm_context), std::move(llvm_module));
}

}  // namespace xla::cpu
