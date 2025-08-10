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

#include "machina/xla/backends/cpu/codegen/jit_compiler.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "toolchain/AsmParser/Parser.h"
#include "toolchain/ExecutionEngine/JITSymbol.h"
#include "toolchain/ExecutionEngine/Orc/AbsoluteSymbols.h"
#include "toolchain/ExecutionEngine/Orc/Core.h"
#include "toolchain/ExecutionEngine/Orc/CoreContainers.h"
#include "toolchain/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "toolchain/ExecutionEngine/Orc/Shared/ExecutorSymbolDef.h"
#include "toolchain/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "toolchain/IR/DataLayout.h"
#include "toolchain/IR/LLVMContext.h"
#include "toolchain/Support/Error.h"
#include "toolchain/Support/SourceMgr.h"
#include "toolchain/Target/TargetMachine.h"
#include "toolchain/Target/TargetOptions.h"
#include "machina/xla/backends/cpu/codegen/ir_compiler.h"
#include "machina/xla/backends/cpu/codegen/kernel_api_ir_builder.h"
#include "machina/xla/backends/cpu/runtime/function_library.h"
#include "machina/xla/tsl/lib/core/status_test_util.h"
#include "machina/xla/tsl/platform/env.h"
#include "machina/xla/tsl/platform/errors.h"
#include "machina/xla/tsl/platform/statusor.h"
#include "machina/xla/tsl/platform/threadpool.h"
#include "machina/xla/util.h"

namespace xla::cpu {

// We use static function to compile the function library, because we transfer
// compiler object into the function and make sure that it gets destroyed before
// returning the function library to the caller, as we test that we don't
// accidentally reference freed objects owned by the compiler.
static absl::StatusOr<std::unique_ptr<FunctionLibrary>> Compile(
    JitCompiler compiler, absl::Span<const FunctionLibrary::Symbol> symbols) {
  return std::move(compiler).Compile(symbols);
};

// Parses the LLVM IR into a ThreadSafeModule.
static absl::StatusOr<toolchain::orc::ThreadSafeModule> ParseModule(
    toolchain::orc::ThreadSafeContext& context, absl::string_view ir,
    absl::string_view name) {
  toolchain::SMDiagnostic diagnostic;
  auto m = context.withContextDo([&](toolchain::LLVMContext* ctxt) {
    toolchain::MemoryBufferRef ir_buffer(ir, name);
    return toolchain::parseAssembly(ir_buffer, diagnostic, *ctxt);
  });
  if (m == nullptr) {
    return Internal("Failed to parse LLVM IR: %s",
                    diagnostic.getMessage().str());
  }

  SetModuleMemoryRegionName(*m, "jit_compiler_test");

  return toolchain::orc::ThreadSafeModule(std::move(m), context);
}

TEST(JitCompilerTest, Compile) {
  auto context = std::make_unique<toolchain::LLVMContext>();
  toolchain::orc::ThreadSafeContext tsc(std::move(context));

  JitCompiler::Options options;
  options.num_dylibs = 2;

  // Use thread pool to run compilation tasks in parallel.
  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "test", 2);
  std::atomic<int32_t> num_tasks = 0;
  JitCompiler::TaskRunner task_runner = [&](JitCompiler::Task task) {
    num_tasks++;
    thread_pool.Schedule(std::move(task));
  };

  std::unique_ptr<IrCompiler> ir_compiler =
      IrCompiler::Create(toolchain::TargetOptions(), IrCompiler::Options(),
                         IrCompiler::CompilationHooks());

  TF_ASSERT_OK_AND_ASSIGN(
      auto compiler,
      JitCompiler::Create(std::move(options), std::move(ir_compiler),
                          std::move(task_runner)));

  constexpr absl::string_view add_in_place_ir = R"(
    define void @AddInplace(ptr %arg) {
      %v0 = load float, ptr %arg
      %v1 = fadd float %v0, %v0
      store float %v1, ptr %arg
      ret void
    })";

  constexpr absl::string_view mul_in_place_ir = R"(
    define void @MulInplace(ptr %arg) {
      %v0 = load float, ptr %arg
      %v1 = fmul float %v0, %v0
      store float %v1, ptr %arg
      ret void
    })";

  auto add_module = [&](absl::string_view ir, absl::string_view name,
                        size_t dylib_index) -> absl::Status {
    TF_ASSIGN_OR_RETURN(toolchain::orc::ThreadSafeModule tsm,
                        ParseModule(tsc, ir, name));
    TF_RETURN_IF_ERROR(compiler.AddModule(std::move(tsm), dylib_index));
    return absl::OkStatus();
  };

  TF_ASSERT_OK(add_module(add_in_place_ir, "AddInplace", 0));
  TF_ASSERT_OK(add_module(mul_in_place_ir, "MulInplace", 1));

  using ScalarFn = void(float*);
  std::vector<FunctionLibrary::Symbol> symbols = {
      FunctionLibrary::Sym<ScalarFn>("AddInplace"),
      FunctionLibrary::Sym<ScalarFn>("MulInplace")};

  TF_ASSERT_OK_AND_ASSIGN(auto function_library,
                          Compile(std::move(compiler), symbols));

  EXPECT_GE(num_tasks, 2);

  TF_ASSERT_OK_AND_ASSIGN(
      ScalarFn * add_in_place,
      function_library->ResolveFunction<ScalarFn>("AddInplace"));

  TF_ASSERT_OK_AND_ASSIGN(
      ScalarFn * mul_in_place,
      function_library->ResolveFunction<ScalarFn>("MulInplace"));

  EXPECT_NE(add_in_place, nullptr);
  EXPECT_NE(mul_in_place, nullptr);

  float value = 1.0f;
  add_in_place(&value);
  EXPECT_EQ(value, 2.0f);

  mul_in_place(&value);
  EXPECT_EQ(value, 4.0f);
}

class ExternalDefinitionGenerator : public toolchain::orc::DefinitionGenerator {
 public:
  static void AddInplace(float* value) { *value += *value; }

  toolchain::Error tryToGenerate(toolchain::orc::LookupState&, toolchain::orc::LookupKind,
                            toolchain::orc::JITDylib& jit_dylib,
                            toolchain::orc::JITDylibLookupFlags,
                            const toolchain::orc::SymbolLookupSet& names) final {
    toolchain::orc::SymbolMap new_defs;
    for (auto& [name, flags] : names) {
      if ((*name).contains("external_fn")) {
        new_defs[name] = toolchain::orc::ExecutorSymbolDef{
            toolchain::orc::ExecutorAddr(reinterpret_cast<uint64_t>(&AddInplace)),
            toolchain::JITSymbolFlags::None};
      }
    }

    cantFail(jit_dylib.define(toolchain::orc::absoluteSymbols(std::move(new_defs))));
    return toolchain::Error::success();
  }
};

TEST(JitCompilerTest, ExternalDefinitionGenerator) {
  auto context = std::make_unique<toolchain::LLVMContext>();
  toolchain::orc::ThreadSafeContext tsc(std::move(context));

  JitCompiler::Options options;
  options.definition_generator = [](const toolchain::DataLayout& data_layout) {
    return std::make_unique<ExternalDefinitionGenerator>();
  };

  std::unique_ptr<IrCompiler> ir_compiler =
      IrCompiler::Create(toolchain::TargetOptions(), IrCompiler::Options(),
                         IrCompiler::CompilationHooks());

  TF_ASSERT_OK_AND_ASSIGN(
      auto compiler,
      JitCompiler::Create(std::move(options), std::move(ir_compiler),
                          /*task_runner=*/nullptr));

  constexpr absl::string_view call_external_fn_ir = R"(
    declare void @__external_fn(ptr %arg)

    define void @CallExternalFn(ptr %arg) {
      call void @__external_fn(ptr %arg)
      ret void
    })";

  TF_ASSERT_OK_AND_ASSIGN(
      toolchain::orc::ThreadSafeModule tsm,
      ParseModule(tsc, call_external_fn_ir, "CallExternalFn"));

  TF_ASSERT_OK(compiler.AddModule(std::move(tsm)));

  using ScalarFn = void(float*);
  std::vector<FunctionLibrary::Symbol> symbols = {
      FunctionLibrary::Sym<ScalarFn>("CallExternalFn")};

  TF_ASSERT_OK_AND_ASSIGN(auto function_library,
                          Compile(std::move(compiler), symbols));

  TF_ASSERT_OK_AND_ASSIGN(
      ScalarFn * call_external_fn,
      function_library->ResolveFunction<ScalarFn>("CallExternalFn"));

  float value = 1.0f;
  call_external_fn(&value);
  EXPECT_EQ(value, 2.0f);
}

}  // namespace xla::cpu
