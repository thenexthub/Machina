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

#include "machina/xla/backends/cpu/codegen/ir_compiler.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/call_once.h"
#include "absl/base/nullability.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "toolchain/ADT/SmallVector.h"
#include "toolchain/Analysis/CGSCCPassManager.h"
#include "toolchain/Analysis/LoopAnalysisManager.h"
#include "toolchain/Analysis/TargetLibraryInfo.h"
#include "toolchain/ExecutionEngine/ExecutionEngine.h"
#include "toolchain/ExecutionEngine/Orc/Mangling.h"
#include "toolchain/IR/LegacyPassManager.h"
#include "toolchain/IR/Metadata.h"
#include "toolchain/IR/PassManager.h"
#include "toolchain/IR/Verifier.h"
#include "toolchain/MC/MCContext.h"
#include "toolchain/Object/ObjectFile.h"
#include "toolchain/Passes/OptimizationLevel.h"
#include "toolchain/Passes/PassBuilder.h"
#include "toolchain/Passes/StandardInstrumentations.h"
#include "toolchain/Support/Casting.h"
#include "toolchain/Support/CodeGen.h"
#include "toolchain/Support/Errc.h"
#include "toolchain/Support/Error.h"
#include "toolchain/Support/MemoryBuffer.h"
#include "toolchain/Support/SmallVectorMemoryBuffer.h"
#include "toolchain/Support/TargetSelect.h"
#include "toolchain/Support/raw_ostream.h"
#include "toolchain/Target/TargetMachine.h"
#include "toolchain/TargetParser/Host.h"
#include "toolchain/TargetParser/Triple.h"
#include "toolchain/Transforms/IPO/AlwaysInliner.h"
#include "toolchain/Transforms/Instrumentation/DataFlowSanitizer.h"
#include "machina/xla/backends/cpu/codegen/cpu_features.h"
#include "machina/xla/backends/cpu/codegen/kernel_api_ir_builder.h"
#include "machina/xla/backends/cpu/codegen/polynomial_approximations.h"
#include "machina/xla/codegen/intrinsic/intrinsic_compiler_lib.h"
#include "machina/xla/codegen/intrinsic_lib.h"
#include "machina/xla/service/cpu/backend_config.pb.h"
#include "machina/xla/service/cpu/cpu_options.h"
#include "machina/xla/service/hlo_module_config.h"
#include "machina/xla/service/llvm_ir/llvm_util.h"
#include "machina/xla/tsl/platform/statusor.h"
#include "machina/xla/util.h"
#include "machina/xla/xla.pb.h"
#include "tsl/platform/cpu_info.h"

namespace xla::cpu {

void SetXlaCpuBackendOptions(toolchain::Module& llvm_module,
                             const LlvmKernelOptions& options) {
  std::vector<std::string> llvm_kernel_options;
  if (options.optimize_for_size()) {
    llvm_kernel_options.emplace_back(options::kXlaOptimizeForSizeCpuOption);
  }
  if (options.disable_loop_unrolling()) {
    llvm_kernel_options.emplace_back(options::kDisableLoopUnrolling);
  }
  if (options.slp_vectorizer_disabled()) {
    llvm_kernel_options.emplace_back(options::kDisableSlpVectorizer);
  }

  toolchain::MDString* options_mdstring = toolchain::MDString::get(
      llvm_module.getContext(), absl::StrJoin(llvm_kernel_options, ","));
  llvm_module.addModuleFlag(toolchain::Module::Error, "xla_backend_extra_options",
                            options_mdstring);
}

static toolchain::OptimizationLevel GetOptimizationLevel(
    IrCompiler::Options options) {
  if (options.optimize_for_size) {
    return toolchain::OptimizationLevel::Os;
  }

  switch (options.opt_level) {
    case toolchain::CodeGenOptLevel::None:
      return toolchain::OptimizationLevel::O0;
    case toolchain::CodeGenOptLevel::Less:
      return toolchain::OptimizationLevel::O1;
    case toolchain::CodeGenOptLevel::Default:
      return toolchain::OptimizationLevel::O2;
    case toolchain::CodeGenOptLevel::Aggressive:
      return toolchain::OptimizationLevel::O3;
  }
}

static std::unique_ptr<HloModuleConfig> ParseXlaBackendExtraOptions(
    absl::string_view config_csv) {
  auto module_config = std::make_unique<HloModuleConfig>();
  DebugOptions& debug_options = module_config->mutable_debug_options();
  auto* map = debug_options.mutable_xla_backend_extra_options();
  std::vector<absl::string_view> vec =
      absl::StrSplit(config_csv, ',', absl::SkipEmpty());
  for (const auto& v : vec) {
    std::vector<absl::string_view> kv = absl::StrSplit(v, '=');
    (*map)[kv[0]] = kv.size() == 1 ? "" : kv[1];
  }
  return module_config;
}

// Returns an HloModuleConfig with its DebugOptions.xla_backend_extra_options
// set by the values embedded in the LLVM module. The rest of the fields
// of the proto should be ignored since they're just the default values.
// We could instead return an unordered_map<str, str>, but we already have
// helpers that expect a DebugOptions, so this ends up being simpler.
static absl_nullable std::unique_ptr<HloModuleConfig> GetXlaBackendExtraOptions(
    const toolchain::Module& llvm_module) {
  toolchain::Metadata* md = llvm_module.getModuleFlag("xla_backend_extra_options");
  if (md == nullptr) return nullptr;
  auto* md_string = toolchain::dyn_cast<toolchain::MDString>(md);
  if (md_string == nullptr) return nullptr;
  std::string config_csv = md_string->getString().str();
  return ParseXlaBackendExtraOptions(config_csv);
}

static toolchain::PipelineTuningOptions GetPipelineTuningOptions(
    const toolchain::Module& module, IrCompiler::Options options,
    const toolchain::TargetMachine* target_machine) {
  auto pto_from_options = [&](const IrCompiler::Options opts) {
    toolchain::PipelineTuningOptions pto;
    pto.LoopVectorization = !opts.optimize_for_size;
    pto.SLPVectorization =
        !opts.optimize_for_size && !opts.disable_slp_vectorizer;
    pto.LoopUnrolling = !opts.disable_loop_unrolling;

    // TODO(b/411125413): Re-enable SLPVectorization once the LLVM bug is fixed.
    pto.SLPVectorization = false;

    return pto;
  };

  std::unique_ptr<HloModuleConfig> config = GetXlaBackendExtraOptions(module);
  if (config == nullptr) {
    return pto_from_options(options);
  }

  // Apply overrides from the embedded config.
  IrCompiler::Options with_overrides(options);
  if (options::OptimizeForSizeRequested(*config)) {
    with_overrides.optimize_for_size = true;
  }
  if (options::SlpVectorizerDisabled(*config)) {
    with_overrides.disable_slp_vectorizer = true;
  }
  if (options::DisableLoopUnrolling(*config)) {
    with_overrides.disable_loop_unrolling = true;
  }
  return pto_from_options(with_overrides);
}

static bool FunctionHasInternalLinkage(const toolchain::Function& function) {
  return function.hasInternalLinkage();
}

std::unique_ptr<IrCompiler> IrCompiler::Create(
    toolchain::TargetOptions target_options, Options options,
    CompilationHooks hooks) {
  TargetMachineBuilder target_machine_builder =
      IrCompiler::InferTargetMachineBuilder(std::move(target_options),
                                            options.opt_level,
                                            options.max_cpu_feature);

  return std::make_unique<IrCompiler>(target_machine_builder,
                                      std::move(options), std::move(hooks));
}

IrCompiler::IrCompiler(TargetMachineBuilder target_machine_builder,
                       Options options, CompilationHooks hooks)
    : IRCompiler(toolchain::orc::IRSymbolMapper::ManglingOptions()),
      target_machine_builder_(std::move(target_machine_builder)),
      options_(std::move(options)),
      hooks_(std::move(hooks)) {}

// Initialize LLVM the first time `InferTargetMachine` is called.
static void InitializeLLVMTarget() {
  toolchain::InitializeNativeTarget();
  toolchain::InitializeNativeTargetAsmPrinter();
}

absl::once_flag initialize_llvm_flag;

absl::StatusOr<std::unique_ptr<toolchain::TargetMachine>>
IrCompiler::InferTargetMachine(
    const toolchain::TargetOptions& target_options, toolchain::CodeGenOptLevel opt_level,
    std::optional<tsl::port::CPUFeature> max_cpu_feature) {
  // Detect machine attributes for the target CPU.
  auto result = DetectMachineAttributes(max_cpu_feature);
  toolchain::SmallVector<std::string> attrs(result.features.begin(),
                                       result.features.end());

  // If `max_cpu_feature` is newer than the host CPU, we should keep the host
  // CPU name, e.g., we don't want to set the target CPU to Skylake when we are
  // on a Broadwell host.
  absl::string_view cpu = result.num_filtered_features
                              ? CpuTargetFromMaxFeature(*max_cpu_feature)
                              : absl::string_view(toolchain::sys::getHostCPUName());

  absl::call_once(initialize_llvm_flag, InitializeLLVMTarget);
  std::unique_ptr<toolchain::TargetMachine> target_machine(
      toolchain::EngineBuilder()
          .setTargetOptions(target_options)
          .setOptLevel(opt_level)
          .selectTarget(
              /*TargetTriple=*/toolchain::Triple(), /*MArch=*/"",
              /*MCPU=*/cpu,
              /*MAttrs=*/attrs));

  if (target_machine == nullptr) {
    return Internal("Failed to create target machine for CPU %s", cpu);
  }

  return std::move(target_machine);
}

IrCompiler::TargetMachineBuilder IrCompiler::InferTargetMachineBuilder(
    const toolchain::TargetOptions& target_options, toolchain::CodeGenOptLevel opt_level,
    std::optional<tsl::port::CPUFeature> max_cpu_feature) {
  return [target_options, opt_level, max_cpu_feature] {
    return InferTargetMachine(target_options, opt_level, max_cpu_feature);
  };
}

toolchain::Expected<std::unique_ptr<toolchain::MemoryBuffer>> IrCompiler::operator()(
    toolchain::Module& module) {
  absl::string_view module_name = module.getName();
  MACHINA_MACHINA_XLA_SCOPED_LOGGING_TIMER_LEVEL(
      absl::StrCat("Compiled LLVM module: ", module_name), 1);

  VLOG(2) << "IR before optimizations";
  MACHINA_MACHINA_XLA_VLOG_LINES(2, llvm_ir::DumpToString(&module));

  // Get a target machine for compilation. If compilations run concurrently on
  // multiple threads, `IrCompiler` user (in most cases `SimpleOrcJIT`)
  // must guarantee that target machine builder will return a unique
  // TargetMachine for each compilation, as it is not thread safe.
  absl::StatusOr<std::unique_ptr<toolchain::TargetMachine>> target_machine =
      build_target_machine();

  if (!target_machine.ok()) {
    return toolchain::make_error<toolchain::StringError>(
        toolchain::errc::invalid_argument,
        absl::StrFormat(
            "Failed to create target machine for IR compilation: %s",
            target_machine.status().message()));
  }

  {  // Synchronize access to user-defined hooks.
    absl::MutexLock lock(&mutex_);
    if (hooks_.pre_optimization) {
      hooks_.pre_optimization(module);
    }
  }

  if (toolchain::Error ir_passes_error =
          RunIrPasses(module, target_machine->get())) {
    return ir_passes_error;
  }

  VLOG(2) << "IR after optimizations";
  MACHINA_MACHINA_XLA_VLOG_LINES(2, llvm_ir::DumpToString(&module));

  {  // Synchronize access to user-defined hooks.
    absl::MutexLock lock(&mutex_);
    if (hooks_.post_optimization) {
      hooks_.post_optimization(module);
    }
  }

  std::unique_ptr<toolchain::MemoryBuffer> mc_memory_buffer =
      EmitMachineCode(module, target_machine->get());

  {  // Synchronize access to user-defined hooks.
    absl::MutexLock lock(&mutex_);
    if (hooks_.post_codegen) {
      toolchain::Expected<std::unique_ptr<toolchain::object::ObjectFile>> obj_file =
          toolchain::object::ObjectFile::createObjectFile(*mc_memory_buffer);
      if (obj_file) {
        hooks_.post_codegen(module, *obj_file.get());
      } else {
        LOG(WARNING) << "Could not convert memory buffer to object file";
      }
    }
  }

  return std::move(mc_memory_buffer);
}

toolchain::Error IrCompiler::RunIrPasses(toolchain::Module& module,
                                    toolchain::TargetMachine* target_machine) const {
  if (absl::c_any_of(module.getFunctionList(), FunctionHasInternalLinkage)) {
    codegen::intrinsic::RunInlineAndOptPasses(module);
  }

  toolchain::PipelineTuningOptions pto =
      GetPipelineTuningOptions(module, options_, target_machine);
  toolchain::LoopAnalysisManager lam;
  toolchain::FunctionAnalysisManager fam;
  toolchain::CGSCCAnalysisManager cgam;
  toolchain::ModuleAnalysisManager mam;

  toolchain::PassInstrumentationCallbacks pic;
  toolchain::StandardInstrumentations si(module.getContext(), false);
  si.registerCallbacks(pic, &mam);

  toolchain::PassBuilder pb(target_machine, pto, {}, &pic);

  // Add the appropriate TargetLibraryInfo.
  toolchain::Triple target_triple(target_machine->getTargetTriple());
  auto target_library_info_impl =
      std::make_unique<toolchain::TargetLibraryInfoImpl>(target_triple);
  target_library_info_impl->addVectorizableFunctions(
      PolynomialApproximationsVectorization());
  codegen::IntrinsicFunctionLib intrinsic_lib(
      target_machine->getTargetFeatureString().str());
  target_library_info_impl->addVectorizableFunctions(
      intrinsic_lib.Vectorizations());

  fam.registerPass(
      [&] { return toolchain::TargetLibraryAnalysis(*target_library_info_impl); });

  pb.registerModuleAnalyses(mam);
  pb.registerCGSCCAnalyses(cgam);
  pb.registerFunctionAnalyses(fam);
  pb.registerLoopAnalyses(lam);
  pb.crossRegisterProxies(lam, fam, cgam, mam);

  toolchain::ModulePassManager pm;

  if (options_.dfsan_enabled) {
    pm.addPass(toolchain::DataFlowSanitizerPass(options_.dfsan_abi_list_files));
  }

  toolchain::OptimizationLevel opt_level = GetOptimizationLevel(options_);
  if (opt_level == toolchain::OptimizationLevel::O0) {
    pm.addPass(pb.buildO0DefaultPipeline(opt_level));
  } else {
    pm.addPass(pb.buildPerModuleDefaultPipeline(opt_level));
  }

  {
    std::string error_string;
    toolchain::raw_string_ostream error_stream(error_string);
    if (toolchain::verifyModule(module, &error_stream)) {
      return toolchain::make_error<toolchain::StringError>(
          toolchain::errc::invalid_argument,
          absl::StrFormat("Invalid LLVM IR before optimizations:\n%s",
                          error_stream.str()));
    }
  }

  pm.run(module, mam);

  {
    std::string error_string;
    toolchain::raw_string_ostream error_stream(error_string);
    if (toolchain::verifyModule(module, &error_stream)) {
      return toolchain::make_error<toolchain::StringError>(
          toolchain::errc::invalid_argument,
          absl::StrFormat("Invalid LLVM IR after optimizations:\n%s\n",
                          error_stream.str()));
    }
  }

  auto replaced_functions = intrinsic_lib.RewriteIntrinsicFunctions(module);
  RewriteToPolynomialApproximations(&module, options_.fast_math_flags);
  if (!replaced_functions.empty()) {
    codegen::intrinsic::RemoveFromCompilerUsed(module, replaced_functions);
    codegen::intrinsic::RunInlineAndOptPasses(module);
  }

  return toolchain::Error::success();
}

std::unique_ptr<toolchain::MemoryBuffer> IrCompiler::EmitMachineCode(
    toolchain::Module& module, toolchain::TargetMachine* target_machine) const {
  // Buffer for holding machine code prior to constructing the ObjectFile.
  toolchain::SmallVector<char, 0> mc_stream_buffer;
  toolchain::raw_svector_ostream ostream(mc_stream_buffer);

  // Generate code.
  toolchain::MCContext* mc_context;
  toolchain::legacy::PassManager codegen_passes;
  target_machine->addPassesToEmitMC(codegen_passes, mc_context, ostream);
  codegen_passes.run(module);

  toolchain::NamedMDNode* memory_region_name_md =
      module.getNamedMetadata(std::string(kMemoryRegionNameMetadataName));
  CHECK(memory_region_name_md != nullptr)
      << "Memory region name metadata not found in LLVM module.";
  CHECK_GT(memory_region_name_md->getNumOperands(), 0);
  toolchain::MDNode* node = memory_region_name_md->getOperand(0);
  CHECK(node != nullptr);
  CHECK_GT(node->getNumOperands(), 0);
  toolchain::MDString* md_str = toolchain::dyn_cast<toolchain::MDString>(node->getOperand(0));
  CHECK(md_str != nullptr);
  toolchain::StringRef mem_region_name_str = md_str->getString();

  return std::make_unique<toolchain::SmallVectorMemoryBuffer>(
      std::move(mc_stream_buffer), mem_region_name_str);
}

toolchain::CodeGenOptLevel IrCompiler::GetCodeGenOptLevel(
    const HloModuleConfig& module_config) {
  switch (module_config.debug_options().xla_backend_optimization_level()) {
    case 1:
      return toolchain::CodeGenOptLevel::Less;
    case 2:
      return toolchain::CodeGenOptLevel::Default;
    case 3:
      return toolchain::CodeGenOptLevel::Aggressive;
    default:
      return toolchain::CodeGenOptLevel::None;
  }
}

absl::StatusOr<std::unique_ptr<toolchain::TargetMachine>>
IrCompiler::build_target_machine() const {
  TF_ASSIGN_OR_RETURN(auto target_machine, target_machine_builder_());

  absl::string_view current_features(target_machine->getTargetFeatureString());

  std::vector<std::string> additional_features;
  for (absl::string_view feature : absl::StrSplit(current_features, ',')) {
    // Scatter & gather can result in very poor performance.
    if (absl::StartsWith(feature, "+avx512")) {
      additional_features.push_back("+prefer-no-scatter");
      additional_features.push_back("+prefer-no-gather");
    }
  }

  if (additional_features.empty()) {
    return target_machine;
  }
  std::string additional_features_str = absl::StrJoin(additional_features, ",");
  if (current_features.empty()) {
    target_machine->setTargetFeatureString(additional_features_str);
  } else {
    target_machine->setTargetFeatureString(
        absl::StrCat(current_features, ",", additional_features_str));
  }

  return target_machine;
}

}  // namespace xla::cpu
