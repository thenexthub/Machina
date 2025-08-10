// Copyright 2020 The TensorFlow Runtime Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//===- hlo_to_kernel.cc -----------------------------------------*- C++ -*-===//
//
// This file implements the entry point to compile a hlo op to a kernel.
//
//===----------------------------------------------------------------------===//
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "toolchain/ADT/SmallString.h"
#include "toolchain/Analysis/TargetLibraryInfo.h"
#include "toolchain/CodeGen/CommandFlags.h"
#include "toolchain/IR/LLVMContext.h"
#include "toolchain/IR/LegacyPassManager.h"
#include "toolchain/IR/Module.h"
#include "toolchain/MC/TargetRegistry.h"
#include "toolchain/Support/CodeGen.h"
#include "toolchain/Support/CommandLine.h"
#include "toolchain/Support/SourceMgr.h"
#include "toolchain/Support/TargetSelect.h"
#include "toolchain/Support/raw_ostream.h"
#include "toolchain/Target/TargetMachine.h"
#include "toolchain/TargetParser/Host.h"
#include "toolchain/TargetParser/Triple.h"
#include "mlir/Dialect/MemRef/Transforms/AllocationOpInterfaceImpl.h"  // part of Codira Toolchain
#include "mlir/ExecutionEngine/OptUtils.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/Diagnostics.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/IR/OwningOpRef.h"  // part of Codira Toolchain
#include "mlir/Pass/PassManager.h"  // part of Codira Toolchain
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"  // part of Codira Toolchain
#include "mlir/Target/LLVMIR/Export.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/init_mlir.h"
#include "machina/compiler/mlir/tools/kernel_gen/kernel_creator.h"
#include "machina/xla/service/llvm_ir/llvm_command_line_options.h"
#include "machina/xla/tsl/platform/errors.h"
#include "machina/xla/tsl/platform/statusor.h"
#include "machina/core/platform/env.h"
#include "machina/core/platform/logging.h"
#include "machina/core/platform/status.h"

namespace machina {
namespace kernel_gen {
namespace {

static toolchain::codegen::RegisterCodeGenFlags CGF;

std::unique_ptr<toolchain::TargetMachine> GetTargetMachine(
    toolchain::StringRef host_triple, toolchain::Module* module) {
  toolchain::Triple triple(module->getTargetTriple());
  if (triple.getTriple().empty()) {
    if (!host_triple.empty()) {
      triple = toolchain::Triple(host_triple);
    } else {
      triple = toolchain::Triple(toolchain::sys::getDefaultTargetTriple());
    }
    module->setTargetTriple(toolchain::Triple(triple.getTriple()));
  }

  std::string error;
  const toolchain::Target* target =
      toolchain::TargetRegistry::lookupTarget("", triple, error);
  if (!target) {
    return nullptr;
  }

  toolchain::TargetOptions target_options =
      toolchain::codegen::InitTargetOptionsFromCodeGenFlags(toolchain::Triple());
  return std::unique_ptr<toolchain::TargetMachine>(target->createTargetMachine(
      triple.str(), "generic", "", target_options, toolchain::Reloc::Model::PIC_));
}

// Compiles the given MLIR module via LLVM into an executable binary format.
absl::StatusOr<std::string> EmitToBinary(toolchain::StringRef host_triple,
                                         mlir::ModuleOp module) {
  // Translate the module.
  toolchain::LLVMContext llvm_context;
  mlir::registerLLVMDialectTranslation(*module->getContext());
  std::unique_ptr<toolchain::Module> llvm_module =
      mlir::translateModuleToLLVMIR(module, llvm_context);

  auto target_machine = GetTargetMachine(host_triple, llvm_module.get());
  llvm_module->setDataLayout(target_machine->createDataLayout());

  // Run LLVM's mid-level optimizer to clean up the IR.
  if (mlir::makeOptimizingTransformer(
          /*optLevel=*/2, /*sizeLevel=*/0,
          target_machine.get())(llvm_module.get())) {
    return absl::InternalError("Failed to run LLVM optimizer passess");
  }

  // Set up the output stream.
  toolchain::SmallString<8> outstr;
  toolchain::raw_svector_ostream ostream(outstr);
  ostream.SetUnbuffered();

  toolchain::legacy::PassManager codegen_passes;
  codegen_passes.add(new toolchain::TargetLibraryInfoWrapperPass(
      toolchain::Triple(llvm_module->getTargetTriple())));

  if (target_machine->addPassesToEmitFile(codegen_passes, ostream, nullptr,
                                          toolchain::CodeGenFileType::ObjectFile,
                                          false)) {
    return absl::InternalError("Failed add passes to emit file");
  }
  codegen_passes.run(*llvm_module);
  return ostream.str().str();
}

absl::Status Run(toolchain::StringRef input_file, toolchain::StringRef output_file,
                 toolchain::StringRef host_triple,
                 toolchain::ArrayRef<std::string> architectures,
                 toolchain::ArrayRef<int64_t> tile_sizes,
                 toolchain::ArrayRef<int64_t> unroll_factors, bool print_ptx,
                 bool print_llvmir, bool enable_ftz, bool index_64bit,
                 bool jit_compile, bool jit_i64_indexed_for_large_tensors) {
  // Read TF code.
  std::string hlo_code;
  TF_RETURN_IF_ERROR(
      ReadFileToString(Env::Default(), input_file.str(), &hlo_code));

  // Compile.
  mlir::DialectRegistry registry;
  mlir::memref::registerAllocationOpInterfaceExternalModels(registry);
  mlir::MLIRContext context(registry);

  toolchain::SourceMgr source_mgr;
  mlir::SourceMgrDiagnosticHandler source_mgr_handler(source_mgr, &context);

  TF_ASSIGN_OR_RETURN(
      mlir::OwningOpRef<mlir::ModuleOp> module,
      GenerateKernelForHloCode(context, hlo_code, architectures, tile_sizes,
                               unroll_factors, print_ptx, print_llvmir,
                               enable_ftz, index_64bit, jit_compile,
                               jit_i64_indexed_for_large_tensors,
                               /*apply_cl_options=*/true));

  // Get binary.
  TF_ASSIGN_OR_RETURN(std::string binary, EmitToBinary(host_triple, *module));

  // Write .a file.
  TF_RETURN_IF_ERROR(
      WriteStringToFile(Env::Default(), output_file.str(), binary));
  return absl::OkStatus();
}

}  // namespace
}  // namespace kernel_gen
}  // namespace machina

int main(int argc, char** argv) {
  toolchain::cl::opt<std::string> input_file("input", toolchain::cl::desc("input file"),
                                        toolchain::cl::value_desc("filename"),
                                        toolchain::cl::init("foo.mlir"));
  toolchain::cl::opt<std::string> output_file(
      "output", toolchain::cl::desc("output file"), toolchain::cl::value_desc("filename"),
      toolchain::cl::init("foo.bin"));
  toolchain::cl::opt<bool> index_64bit("index_64bit",
                                  toolchain::cl::desc("enable 64 bit indexing"),
                                  toolchain::cl::init(false));
  toolchain::cl::opt<bool> print_ptx(
      "print-ptx",
      toolchain::cl::desc("print generated PTX code per target architecture."),
      toolchain::cl::init(false));
  toolchain::cl::opt<bool> print_llvmir(
      "print-llvmir", toolchain::cl::desc("print toolchain ir during lowering to ptx."),
      toolchain::cl::init(false));
  toolchain::cl::opt<bool> enable_ftz(
      "enable_ftz",
      toolchain::cl::desc(
          "enable the denormal flush to zero mode when generating code."),
      toolchain::cl::init(false));
  toolchain::cl::opt<bool> jit_compile(
      "jit", toolchain::cl::desc("Generate only a JIT compiler invocation."),
      toolchain::cl::init(false));
  toolchain::cl::opt<std::string> host_triple(
      "host-triple", toolchain::cl::desc("Override host triple for module"));
  toolchain::cl::list<std::string> architectures(
      "arch", toolchain::cl::desc("target architectures (e.g. sm_70 or compute_75)"),
      toolchain::cl::ZeroOrMore, toolchain::cl::CommaSeparated);
  toolchain::cl::list<int64_t> tile_sizes(
      "tile_sizes", toolchain::cl::desc("tile sizes to use"), toolchain::cl::ZeroOrMore,
      toolchain::cl::CommaSeparated);
  toolchain::cl::list<int64_t> unroll_factors(
      "unroll_factors",
      toolchain::cl::desc("factors to unroll by, separated by commas"),
      toolchain::cl::ZeroOrMore, toolchain::cl::CommaSeparated);
  toolchain::cl::opt<bool> jit_i64_indexed_for_large_tensors(
      "jit_i64_indexed_for_large_tensors",
      toolchain::cl::desc(
          "Enable JIT compilation of i64-indexed kernels for large inputs."),
      toolchain::cl::init(false));

  machina::InitMlir y(&argc, &argv);

#ifdef TF_LLVM_X86_AVAILABLE
  LLVMInitializeX86Target();
  LLVMInitializeX86TargetInfo();
  LLVMInitializeX86TargetMC();
  LLVMInitializeX86AsmPrinter();
#endif

#ifdef TF_LLVM_AARCH64_AVAILABLE
  LLVMInitializeAArch64Target();
  LLVMInitializeAArch64TargetInfo();
  LLVMInitializeAArch64TargetMC();
  LLVMInitializeAArch64AsmPrinter();
#endif

  mlir::registerPassManagerCLOptions();
  mlir::registerMLIRContextCLOptions();

  // Forward cli options to XLA, as it will reset toolchain options internally
  // during the first invocation.
  auto& xla_llvm_global_options =
      xla::llvm_ir::LLVMCommandLineOptionsLock::GetGlobalOptions();
  xla_llvm_global_options.insert(xla_llvm_global_options.end(), argv + 1,
                                 argv + argc);
  toolchain::cl::ParseCommandLineOptions(argc, argv, "TF op kernel generator\n");

  auto status = machina::kernel_gen::Run(
      input_file, output_file, host_triple, architectures, tile_sizes,
      unroll_factors, print_ptx, print_llvmir, enable_ftz, index_64bit,
      jit_compile, jit_i64_indexed_for_large_tensors);
  if (!status.ok()) {
    LOG(ERROR) << status;
    return 1;
  }
  return 0;
}
