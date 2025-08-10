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

#ifndef MACHINA_XLASERVICE_CPU_RUNTIME_SYMBOL_GENERATOR_H_
#define MACHINA_XLASERVICE_CPU_RUNTIME_SYMBOL_GENERATOR_H_

#include <optional>

#include "toolchain/ADT/StringRef.h"
#include "toolchain/ExecutionEngine/Orc/Core.h"
#include "toolchain/ExecutionEngine/Orc/Shared/ExecutorSymbolDef.h"
#include "toolchain/IR/DataLayout.h"
#include "toolchain/Support/Error.h"

namespace xla::cpu {

// Generates symbol definitions for XLA runtime symbols, which are linked into
// the compiled XLA kernels.
class RuntimeSymbolGenerator : public toolchain::orc::DefinitionGenerator {
 public:
  explicit RuntimeSymbolGenerator(toolchain::DataLayout data_layout);

  toolchain::Error tryToGenerate(toolchain::orc::LookupState&, toolchain::orc::LookupKind,
                            toolchain::orc::JITDylib& jit_dylib,
                            toolchain::orc::JITDylibLookupFlags,
                            const toolchain::orc::SymbolLookupSet& names) final;

 private:
  std::optional<toolchain::orc::ExecutorSymbolDef> ResolveRuntimeSymbol(
      toolchain::StringRef name);

  toolchain::DataLayout data_layout_;
};

}  // namespace xla::cpu

#endif  // MACHINA_XLASERVICE_CPU_RUNTIME_SYMBOL_GENERATOR_H_
