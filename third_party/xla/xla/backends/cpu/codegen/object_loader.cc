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

#include "machina/xla/backends/cpu/codegen/object_loader.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "toolchain/ADT/SmallVector.h"
#include "toolchain/ADT/StringRef.h"
#include "toolchain/ExecutionEngine/Orc/Core.h"
#include "toolchain/ExecutionEngine/Orc/CoreContainers.h"
#include "toolchain/ExecutionEngine/Orc/ExecutorProcessControl.h"
#include "toolchain/ExecutionEngine/Orc/InProcessMemoryAccess.h"
#include "toolchain/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "toolchain/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "toolchain/ExecutionEngine/Orc/Shared/ExecutorSymbolDef.h"
#include "toolchain/ExecutionEngine/Orc/SymbolStringPool.h"
#include "toolchain/ExecutionEngine/Orc/TaskDispatch.h"
#include "toolchain/IR/DataLayout.h"
#include "toolchain/IR/Mangler.h"
#include "toolchain/Support/Error.h"
#include "toolchain/Support/ErrorHandling.h"
#include "toolchain/Support/MemoryBuffer.h"
#include "machina/xla/backends/cpu/codegen/compiled_function_library.h"
#include "machina/xla/backends/cpu/codegen/execution_engine.h"
#include "machina/xla/backends/cpu/runtime/function_library.h"
#include "machina/xla/tsl/platform/statusor.h"

namespace xla::cpu {

namespace {
// TODO: move to ExecutorProcessControl-based APIs.
class UnsupportedExecutorProcessControl
    : public toolchain::orc::ExecutorProcessControl,
      private toolchain::orc::InProcessMemoryAccess {
 public:
  UnsupportedExecutorProcessControl()
      : ExecutorProcessControl(
            std::make_shared<toolchain::orc::SymbolStringPool>(),
            std::make_unique<toolchain::orc::InPlaceTaskDispatcher>()),
        InProcessMemoryAccess(toolchain::Triple("").isArch64Bit()) {
    this->TargetTriple = toolchain::Triple("");
    this->MemAccess = this;
  }

  toolchain::Expected<int32_t> runAsMain(toolchain::orc::ExecutorAddr MainFnAddr,
                                    toolchain::ArrayRef<std::string> Args) override {
    llvm_unreachable("Unsupported");
  }

  toolchain::Expected<int32_t> runAsVoidFunction(
      toolchain::orc::ExecutorAddr VoidFnAddr) override {
    llvm_unreachable("Unsupported");
  }

  toolchain::Expected<int32_t> runAsIntFunction(toolchain::orc::ExecutorAddr IntFnAddr,
                                           int Arg) override {
    llvm_unreachable("Unsupported");
  }

  void callWrapperAsync(toolchain::orc::ExecutorAddr WrapperFnAddr,
                        IncomingWFRHandler OnComplete,
                        toolchain::ArrayRef<char> ArgBuffer) override {
    llvm_unreachable("Unsupported");
  }

  toolchain::Error disconnect() override { return toolchain::Error::success(); }
};
}  // namespace

static std::unique_ptr<ExecutionEngine> CreateExecutionEngine(
    const toolchain::DataLayout& data_layout,
    ExecutionEngine::DefinitionGenerator definition_generator) {
  return std::make_unique<ExecutionEngine>(
      std::make_unique<toolchain::orc::ExecutionSession>(
          std::make_unique<UnsupportedExecutorProcessControl>()),
      data_layout, definition_generator);
}

ObjectLoader::ObjectLoader(
    size_t num_dylibs, const toolchain::DataLayout& data_layout,
    ExecutionEngine::DefinitionGenerator definition_generator)
    : execution_engine_(
          CreateExecutionEngine(data_layout, definition_generator)) {
  execution_engine_->AllocateDylibs(std::max<size_t>(1, num_dylibs));
}

ObjectLoader::ObjectLoader(std::unique_ptr<ExecutionEngine> execution_engine)
    : execution_engine_(std::move(execution_engine)) {}

absl::Status ObjectLoader::AddObjFile(const std::string& obj_file,
                                      const std::string& memory_buffer_name,
                                      size_t dylib_index) {
  toolchain::StringRef data(obj_file.data(), obj_file.size());

  auto obj_file_mem_buffer =
      toolchain::MemoryBuffer::getMemBuffer(data, memory_buffer_name);

  return AddObjFile(std::move(obj_file_mem_buffer), dylib_index);
}

absl::Status ObjectLoader::AddObjFile(
    std::unique_ptr<toolchain::MemoryBuffer> obj_file, size_t dylib_index) {
  if (dylib_index >= num_dylibs()) {
    return absl::Status(
        absl::StatusCode::kInvalidArgument,
        absl::StrFormat("Invalid dylib index %d (num dylibs: %d))", dylib_index,
                        num_dylibs()));
  }

  if (!obj_file) {
    return absl::Status(absl::StatusCode::kInvalidArgument,
                        "Failed to create memory buffer");
  }

  TF_ASSIGN_OR_RETURN(toolchain::orc::JITDylib * dylib,
                      execution_engine_->dylib(dylib_index));
  if (auto err =
          execution_engine_->object_layer()->add(*dylib, std::move(obj_file))) {
    return absl::Status(
        absl::StatusCode::kInvalidArgument,
        absl::StrFormat("Failed to add object file to dylib %d: %s",
                        dylib_index, toolchain::toString(std::move(err))));
  }

  return absl::OkStatus();
}

std::function<std::string(absl::string_view)> ObjectLoader::GetMangler() {
  // Mangle symbol names for the target machine data layout.
  auto mangle = [this](absl::string_view name) {
    toolchain::SmallVector<char, 40> mangled;
    toolchain::Mangler::getNameWithPrefix(mangled, name,
                                     this->execution_engine_->data_layout());
    return std::string(mangled.begin(), mangled.end());
  };
  return mangle;
}

absl::StatusOr<toolchain::orc::SymbolMap> ObjectLoader::LookupSymbols(
    absl::Span<const Symbol> symbols) {
  // Mangle symbol names for the target machine data layout.
  auto mangle = GetMangler();

  // Build a symbol lookup set.
  toolchain::orc::SymbolLookupSet lookup_set;
  for (const auto& symbol : symbols) {
    VLOG(5) << absl::StreamFormat(" - look up symbol: %s", symbol.name);
    lookup_set.add(
        execution_engine_->execution_session()->intern(mangle(symbol.name)));
  }

  // Build a search order for the dynamic libraries.
  toolchain::orc::JITDylibSearchOrder search_order(num_dylibs());
  for (size_t i = 0; i < num_dylibs(); ++i) {
    TF_ASSIGN_OR_RETURN(toolchain::orc::JITDylib * dylib,
                        execution_engine_->dylib(i));
    search_order[i] = std::make_pair(
        dylib, toolchain::orc::JITDylibLookupFlags::MatchExportedSymbolsOnly);
  }

  // Look up all requested symbols in the execution session.
  auto symbol_map = execution_session()->lookup(std::move(search_order),
                                                std::move(lookup_set));

  if (auto err = symbol_map.takeError()) {
    return absl::Status(absl::StatusCode::kInternal,
                        absl::StrFormat("%s", toolchain::toString(std::move(err))));
  }

  return symbol_map.get();
}

absl::StatusOr<std::unique_ptr<FunctionLibrary>>
ObjectLoader::CreateFunctionLibrary(absl::Span<const Symbol> symbols,
                                    toolchain::orc::SymbolMap& symbol_map) && {
  auto mangle = GetMangler();

  // Resolve type-erased symbol pointers from the symbol map.
  using ResolvedSymbol = CompiledFunctionLibrary::ResolvedSymbol;
  absl::flat_hash_map<std::string, ResolvedSymbol> resolved_map;

  for (const auto& symbol : symbols) {
    auto symbol_name = execution_session()->intern(mangle(symbol.name));
    toolchain::orc::ExecutorSymbolDef symbol_def = symbol_map.at(symbol_name);
    toolchain::orc::ExecutorAddr symbol_addr = symbol_def.getAddress();
    void* ptr = reinterpret_cast<void*>(symbol_addr.getValue());
    resolved_map[symbol.name] = ResolvedSymbol{symbol.type_id, ptr};
  }

  return std::make_unique<CompiledFunctionLibrary>(std::move(execution_engine_),
                                                   std::move(resolved_map));
}

absl::StatusOr<std::unique_ptr<FunctionLibrary>> ObjectLoader::Load(
    absl::Span<const Symbol> symbols) && {
  TF_ASSIGN_OR_RETURN(auto symbol_map, LookupSymbols(symbols));
  return std::move(*this).CreateFunctionLibrary(symbols, symbol_map);
}

}  // namespace xla::cpu
