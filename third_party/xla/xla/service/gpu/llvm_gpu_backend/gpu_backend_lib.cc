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

#include "machina/xla/service/gpu/llvm_gpu_backend/gpu_backend_lib.h"

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <system_error>  // NOLINT
#include <utility>
#include <variant>
#include <vector>

#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "toolchain/ADT/Any.h"
#include "toolchain/ADT/StringSet.h"
#include "toolchain/Analysis/CGSCCPassManager.h"
#include "toolchain/Analysis/LazyCallGraph.h"
#include "toolchain/Analysis/LoopAnalysisManager.h"
#include "toolchain/Analysis/TargetTransformInfo.h"
#include "toolchain/Bitcode/BitcodeReader.h"
#include "toolchain/Bitcode/BitcodeWriter.h"
#include "toolchain/CodeGen/CommandFlags.h"
#include "toolchain/IR/LLVMContext.h"
#include "toolchain/IR/LegacyPassManager.h"
#include "toolchain/IR/Metadata.h"
#include "toolchain/IR/Module.h"
#include "toolchain/IR/PassManager.h"
#include "toolchain/IR/Verifier.h"
#include "toolchain/InitializePasses.h"
#include "toolchain/Linker/Linker.h"
#include "toolchain/MC/TargetRegistry.h"
#include "toolchain/PassRegistry.h"
#include "toolchain/Passes/OptimizationLevel.h"
#include "toolchain/Passes/PassBuilder.h"
#include "toolchain/Passes/StandardInstrumentations.h"
#include "toolchain/Support/CodeGen.h"
#include "toolchain/Support/raw_ostream.h"
#include "toolchain/Target/TargetMachine.h"
#include "toolchain/Transforms/IPO/AlwaysInliner.h"
#include "toolchain/Transforms/IPO/Internalize.h"
#include "toolchain/Transforms/Scalar.h"
#include "machina/xla/service/gpu/llvm_gpu_backend/load_ir_module.h"
#include "machina/xla/service/gpu/llvm_gpu_backend/utils.h"
#include "machina/xla/service/llvm_ir/llvm_type_conversion_util.h"
#include "machina/xla/stream_executor/device_description.h"
#include "machina/xla/tsl/platform/env.h"
#include "machina/xla/tsl/platform/errors.h"
#include "machina/xla/util.h"
#include "machina/xla/xla.pb.h"
#include "tsl/platform/path.h"
#include "tsl/profiler/lib/scoped_annotation.h"

namespace xla {
namespace gpu {

namespace {
static toolchain::codegen::RegisterCodeGenFlags CGF;
}

// Initializes LLVM passes. Uses the PassRegistry mechanism.
void InitializePasses(toolchain::PassRegistry* pass_registry) {
  toolchain::initializeCore(*pass_registry);
  toolchain::initializeCodeGen(*pass_registry);
  toolchain::initializeScalarOpts(*pass_registry);
  toolchain::initializeVectorization(*pass_registry);
  toolchain::initializeIPO(*pass_registry);
  toolchain::initializeAnalysis(*pass_registry);
  toolchain::initializeTransformUtils(*pass_registry);
  toolchain::initializeInstCombine(*pass_registry);
  toolchain::initializeTarget(*pass_registry);
  toolchain::initializeCodeGenPrepareLegacyPassPass(*pass_registry);
}

// Returns the TargetMachine, given a triple.
std::unique_ptr<toolchain::TargetMachine> GetTargetMachine(
    toolchain::Triple triple, absl::string_view cpu_name,
    const DebugOptions& debug_options, absl::string_view feature_str) {
  std::string error;
  const toolchain::Target* target =
      toolchain::TargetRegistry::lookupTarget("", triple, error);
  if (target == nullptr) {
    LOG(FATAL) << "Unable to find Target for triple '" << triple.str() << "'"
               << " -- " << error;
    return nullptr;
  }

  toolchain::TargetOptions target_options =
      toolchain::codegen::InitTargetOptionsFromCodeGenFlags(toolchain::Triple());

  // Set the verbose assembly options.
  target_options.MCOptions.AsmVerbose = false;

  // The selection of codegen optimization level is copied from function
  // GetCodeGenOptLevel in //third_party/toolchain/toolchain/tools/opt/opt.cpp.
  toolchain::CodeGenOptLevel codegen_opt_level;
  switch (debug_options.xla_backend_optimization_level()) {
    case 1:
      codegen_opt_level = toolchain::CodeGenOptLevel::Less;
      break;
    case 2:
      codegen_opt_level = toolchain::CodeGenOptLevel::Default;
      break;
    case 3:
      codegen_opt_level = toolchain::CodeGenOptLevel::Aggressive;
      break;
    default:
      codegen_opt_level = toolchain::CodeGenOptLevel::None;
  }
  return absl::WrapUnique(target->createTargetMachine(
      triple.str(), llvm_ir::AsStringRef(cpu_name),
      llvm_ir::AsStringRef(feature_str), target_options,
      toolchain::codegen::getExplicitRelocModel(),
      toolchain::codegen::getExplicitCodeModel(), codegen_opt_level));
}

// Returns whether the module could use any device bitcode library functions.
bool CouldNeedDeviceBitcode(const toolchain::Module& module) {
  for (const toolchain::Function& function : module.functions()) {
    // The list of prefixes should be in sync with library functions used in
    // target_util.cc.
    if (!function.isIntrinsic() && function.isDeclaration() &&
        (function.getName().starts_with("__nv_") ||
         function.getName().starts_with("__ocml_") ||
         function.getName().starts_with("__ockl_"))) {
      return true;
    }
  }
  return false;
}

// Links the module with a vector of path to bitcode modules.
// The caller must guarantee that the paths exist.
absl::Status LinkWithBitcodeVector(
    toolchain::Module* module, const std::vector<std::string>& bitcode_path_vector) {
  toolchain::Linker linker(*module);

  for (auto& bitcode_path : bitcode_path_vector) {
    if (!tsl::Env::Default()->FileExists(bitcode_path).ok()) {
      LOG(ERROR) << "bitcode module is required by this HLO module but was "
                    "not found at "
                 << bitcode_path;
      return xla::Internal("bitcode module not found at %s", bitcode_path);
    }

    std::unique_ptr<toolchain::Module> bitcode_module =
        LoadIRModule(bitcode_path, &module->getContext());
    // Ignore the data layout of the module we're importing. This avoids a
    // warning from the linker.
    bitcode_module->setDataLayout(module->getDataLayout());
    if (linker.linkInModule(
            std::move(bitcode_module), toolchain::Linker::Flags::LinkOnlyNeeded,
            [](toolchain::Module& M, const toolchain::StringSet<>& GVS) {
              internalizeModule(M, [&GVS](const toolchain::GlobalValue& GV) {
                return !GV.hasName() || (GVS.count(GV.getName()) == 0);
              });
            })) {
      return xla::Internal("Error linking bitcode module from %s",
                           bitcode_path);
    }
  }
  return absl::OkStatus();
}

namespace {

// NOLINTBEGIN: clang-diagnostic-unused-function
// Convenience function for producing a name of a temporary compilation product
// from the input filename.
std::string MakeNameForTempProduct(absl::string_view input_filename,
                                   absl::string_view extension) {
  return ReplaceFilenameExtension(tsl::io::Basename(input_filename), extension);
}
// NOLINTEND: clang-diagnostic-unused-function

void DumpModule(const std::string output_filename, const toolchain::Module* module) {
  std::string content;
  toolchain::raw_string_ostream string_stream(content);
  module->print(string_stream, /*AAW=*/nullptr);

  auto status =
      WriteStringToFile(tsl::Env::Default(), output_filename, content);
  if (!status.ok()) {
    LOG(FATAL) << "Unable to write " << output_filename
               << " to dump LLVM IR: " << status.message();
  }
}

const toolchain::Module* GetModule(toolchain::Any IR) {
  if (const auto** M = toolchain::any_cast<const toolchain::Module*>(&IR)) return *M;

  if (const auto** F = toolchain::any_cast<const toolchain::Function*>(&IR)) {
    return (*F)->getParent();
  }

  if (const auto** C = toolchain::any_cast<const toolchain::LazyCallGraph::SCC*>(&IR)) {
    return (*C)->begin()->getFunction().getParent();
  }

  if (const auto** L = toolchain::any_cast<const toolchain::Loop*>(&IR)) {
    const toolchain::Function* F = (*L)->getHeader()->getParent();
    return F->getParent();
  }

  return nullptr;
}

auto DumpCallbackForModule(std::string module_identifier,
                           std::string outputs_dir) {
  int i = 0;
  return [=](toolchain::StringRef pass, toolchain::Any ir) mutable {
    const toolchain::Module* module = GetModule(ir);
    if (!module) {
      return;
    }

    const std::string basename = ReplaceFilenameExtension(
        absl::string_view(tsl::io::Basename(module_identifier)),
        absl::StrFormat("pass-%03d.before.%s.ll", i++,
                        absl::string_view(pass.str())));
    DumpModule(tsl::io::JoinPath(outputs_dir, basename), module);
  };
}

}  // namespace

absl::Status LinkAndOptimizeModule(
    toolchain::Module* module, se::GpuComputeCapability gpu_version,
    const DebugOptions& debug_options, const std::string& device_bitcode_path,
    TargetModuleLinker module_linker, toolchain::Triple default_target_triple,
    toolchain::TargetMachine* target_machine, int inline_threshold) {
  tsl::profiler::ScopedAnnotation annotation([&] {
    return absl::StrFormat("XlaOptimizeLlvmIr:#module=%s#",
                           module->getName().str());
  });
  TF_RETURN_IF_ERROR(
      module_linker(module, gpu_version, debug_options, device_bitcode_path));

  toolchain::LoopAnalysisManager lam;
  toolchain::FunctionAnalysisManager fam;
  toolchain::CGSCCAnalysisManager cgam;
  toolchain::ModuleAnalysisManager mam;

  if (target_machine) {
    fam.registerPass([&] { return target_machine->getTargetIRAnalysis(); });
  }

  toolchain::PipelineTuningOptions pto;
  pto.SLPVectorization = true;
  pto.InlinerThreshold = inline_threshold;

  toolchain::PassInstrumentationCallbacks pic;

  toolchain::StandardInstrumentations si(module->getContext(), false);
  si.registerCallbacks(pic, &mam);

  toolchain::PassBuilder pb(target_machine, pto, std::nullopt, &pic);
  pb.registerModuleAnalyses(mam);
  pb.registerCGSCCAnalyses(cgam);
  pb.registerFunctionAnalyses(fam);
  pb.registerLoopAnalyses(lam);
  pb.crossRegisterProxies(lam, fam, cgam, mam);

  if (debug_options.xla_gpu_dump_llvmir()) {
    std::string outputs_dir;
    if (!tsl::io::GetTestUndeclaredOutputsDir(&outputs_dir)) {
      outputs_dir = debug_options.xla_dump_to();
    }
    if (!outputs_dir.empty()) {
      pic.registerBeforeNonSkippedPassCallback(
          DumpCallbackForModule(module->getModuleIdentifier(), outputs_dir));
    } else {
      LOG(ERROR) << "--xla_gpu_dump_llvmir is set, but neither the environment "
                 << "variable TEST_UNDECLARED_OUTPUTS_DIR nor the flag "
                 << "--xla_dump_to is set, so the toolchain dumps are disabled.";
    }
  }

  toolchain::OptimizationLevel ol;
  switch (debug_options.xla_backend_optimization_level()) {
    case 0:
      ol = toolchain::OptimizationLevel::O0;
      break;
    case 1:
      ol = toolchain::OptimizationLevel::O1;
      break;
    case 2:
      ol = toolchain::OptimizationLevel::O2;
      break;
    case 3:
      ol = toolchain::OptimizationLevel::O3;
      break;
  }

  toolchain::ModulePassManager mpm;
  mpm.addPass(toolchain::VerifierPass());
  if (ol == toolchain::OptimizationLevel::O0) {
    mpm.addPass(pb.buildO0DefaultPipeline(ol));
  } else {
    mpm.addPass(pb.buildPerModuleDefaultPipeline(ol));
  }
  mpm.addPass(toolchain::VerifierPass());

  mpm.run(*module, mam);

  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace xla
