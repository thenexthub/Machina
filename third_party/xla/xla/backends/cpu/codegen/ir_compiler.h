/* Copyright 2017 The OpenXLA Authors.

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

#ifndef MACHINA_MACHINA_XLA_BACKENDS_CPU_CODEGEN_IR_COMPILER_H_
#define MACHINA_MACHINA_XLA_BACKENDS_CPU_CODEGEN_IR_COMPILER_H_

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "toolchain/ExecutionEngine/Orc/IRCompileLayer.h"
#include "toolchain/IR/FMF.h"
#include "toolchain/IR/LegacyPassManager.h"
#include "toolchain/IR/Module.h"
#include "toolchain/Object/ObjectFile.h"
#include "toolchain/Support/CodeGen.h"
#include "toolchain/Support/Error.h"
#include "toolchain/Target/TargetMachine.h"
#include "toolchain/Target/TargetOptions.h"
#include "machina/xla/service/cpu/backend_config.pb.h"
#include "machina/xla/service/hlo_module_config.h"
#include "tsl/platform/cpu_info.h"

namespace xla::cpu {

void SetXlaCpuBackendOptions(toolchain::Module& llvm_module,
                             const LlvmKernelOptions& options);

// IrCompiler compiles LLVM modules to object files using LLVM compilation
// pipeline customized for XLA:CPU. Default LLVM compilation pipeline is
// optimized for compiling LLVM IR produced by Clang, and in XLA we are a lot
// more constrained and produce a very different IR.
class IrCompiler : public toolchain::orc::IRCompileLayer::IRCompiler {
 public:
  // Returns an instance of `toolchain::TargetMachine` for a compilation. It can be
  // a shared `toolchain::TargetMachine` if compilation is single threaded, or must
  // be a unique instance of `toolchain::TargetMachine` if compilation is multi
  // threaded (because `toolchain::TargetMachine` is not thread safe).
  //
  // See `toolchain::orc::ConcurrentIRCompiler` to see corresponding API in ORC.
  using TargetMachineBuilder =
      std::function<absl::StatusOr<std::unique_ptr<toolchain::TargetMachine>>()>;

  // Options for configuring the LLVM compilation pipeline and optimizations.
  struct Options {
    toolchain::CodeGenOptLevel opt_level = toolchain::CodeGenOptLevel::None;
    bool optimize_for_size = false;

    // Maximum CPU instruction set for wich the compiler should generate code.
    // If instruction set is empty, compiler will generate code for all ISA
    // extensions detected on the current machine.
    std::optional<tsl::port::CPUFeature> max_cpu_feature;

    toolchain::FastMathFlags fast_math_flags;

    bool disable_expensive_passes = false;
    bool disable_slp_vectorizer = false;

    bool disable_loop_unrolling = false;

    bool dfsan_enabled = false;
    std::vector<std::string> dfsan_abi_list_files;
  };

  // Compilation hooks for intercepting IR compilation stages.
  struct CompilationHooks {
    std::function<void(const toolchain::Module&)> pre_optimization;
    std::function<void(const toolchain::Module&)> post_optimization;
    std::function<void(const toolchain::Module&, const toolchain::object::ObjectFile&)>
        post_codegen;
  };

  static std::unique_ptr<IrCompiler> Create(toolchain::TargetOptions target_options,
                                            Options options,
                                            CompilationHooks hooks);

  IrCompiler(TargetMachineBuilder target_machine_builder, Options options,
             CompilationHooks hooks);

  // Infers the `toolchain::TargetMachine` for the current host. If `max_cpu_feature`
  // is provided, it will be used to constrain the set of features that LLVM
  // codegen (instruction selection) is allowed to use, e.g. it can be used to
  // explicitly disable certain AVX512 extensions, in case the compiled
  // executable will be serialized and later loaded on a different machine.
  static absl::StatusOr<std::unique_ptr<toolchain::TargetMachine>>
  InferTargetMachine(const toolchain::TargetOptions& target_options,
                     toolchain::CodeGenOptLevel opt_level,
                     std::optional<tsl::port::CPUFeature> max_cpu_feature);

  // Returns a target machine builder that uses `InferTargetMachine` defined
  // above to infer the target machine for the given options.
  static TargetMachineBuilder InferTargetMachineBuilder(
      const toolchain::TargetOptions& target_options,
      toolchain::CodeGenOptLevel opt_level,
      std::optional<tsl::port::CPUFeature> max_cpu_feature);

  // Compiles a `module` to an ObjectFile.
  toolchain::Expected<std::unique_ptr<toolchain::MemoryBuffer>> operator()(
      toolchain::Module& module) final;

  // Runs the IR passes on the given module.
  toolchain::Error RunIrPasses(toolchain::Module& module,
                          toolchain::TargetMachine* target_machine) const;

  // Emits machine code for the given module.
  std::unique_ptr<toolchain::MemoryBuffer> EmitMachineCode(
      toolchain::Module& module, toolchain::TargetMachine* target_machine) const;

  static toolchain::CodeGenOptLevel GetCodeGenOptLevel(
      const HloModuleConfig& module_config);

  // Build the target machine and add any target specific features.
  absl::StatusOr<std::unique_ptr<toolchain::TargetMachine>> build_target_machine()
      const;

 private:
  TargetMachineBuilder target_machine_builder_;
  Options options_;

  // IRCompiler can be called in concurrently when JitCompiler compiles multiple
  // modules concurrently, we need to make sure that we don't introduce data
  // races when calling user provided compilation hooks.
  absl::Mutex mutex_;
  CompilationHooks hooks_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace xla::cpu

#endif  // MACHINA_MACHINA_XLA_BACKENDS_CPU_CODEGEN_IR_COMPILER_H_
