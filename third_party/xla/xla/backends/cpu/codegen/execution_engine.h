/* Copyright 2025 The OpenXLA Authors.

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

#ifndef MACHINA_MACHINA_XLA_BACKENDS_CPU_CODEGEN_EXECUTION_ENGINE_H_
#define MACHINA_MACHINA_XLA_BACKENDS_CPU_CODEGEN_EXECUTION_ENGINE_H_

#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "toolchain/ExecutionEngine/JITEventListener.h"
#include "toolchain/ExecutionEngine/Orc/Core.h"
#include "toolchain/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "toolchain/IR/DataLayout.h"

namespace xla::cpu {

class ExecutionEngine {
 public:
  // A callback that returns a definition generator that will be added to all
  // dynamic libraries created by the engine. Definition generator enables
  // linking host runtime symbols into the jit-compiled function library.
  using DefinitionGenerator =
      std::function<std::unique_ptr<toolchain::orc::DefinitionGenerator>(
          const toolchain::DataLayout&)>;

  // Specifying a data layout adds a runtime symbol generator to each dylib.
  explicit ExecutionEngine(
      std::unique_ptr<toolchain::orc::ExecutionSession> execution_session,
      const toolchain::DataLayout& data_layout,
      DefinitionGenerator definition_generator = nullptr);

  ExecutionEngine(ExecutionEngine&& other) noexcept = default;
  ExecutionEngine& operator=(ExecutionEngine&& other) = default;

  void AllocateDylibs(size_t num_dylibs);

  void RegisterJITEventListeners();

  // Implementation from LLJIT, required to find symbols on Windows.
  void SetObjectLayerFlags();

  toolchain::orc::RTDyldObjectLinkingLayer* object_layer() {
    return object_layer_.get();
  }

  toolchain::orc::ExecutionSession* execution_session() {
    return execution_session_.get();
  }

  toolchain::DataLayout data_layout() const { return data_layout_; }

  size_t num_dylibs() const { return dylibs_.size(); }

  absl::StatusOr<toolchain::orc::JITDylib*> dylib(size_t dylib_index) {
    if (dylib_index >= num_dylibs()) {
      return absl::Status(
          absl::StatusCode::kInvalidArgument,
          absl::StrFormat("Invalid dylib index %d (num dylibs: %d))",
                          dylib_index, num_dylibs()));
    }
    return dylibs_[dylib_index];
  }

  ~ExecutionEngine();

 private:
  // LLVM execution session that holds jit-compiled functions.
  std::unique_ptr<toolchain::orc::ExecutionSession> execution_session_;
  // Owns resources required for the execution session.
  std::unique_ptr<toolchain::orc::RTDyldObjectLinkingLayer> object_layer_;
  // Non-owning pointers to dynamic libraries created for the execution session.
  std::vector<toolchain::orc::JITDylib*> dylibs_;

  toolchain::DataLayout data_layout_;
  DefinitionGenerator definition_generator_;

  // GDB notification listener (not owned).
  toolchain::JITEventListener* gdb_listener_ = nullptr;
  // Perf notification listener (not owned).
  toolchain::JITEventListener* perf_listener_ = nullptr;
};

}  // namespace xla::cpu

#endif  // MACHINA_MACHINA_XLA_BACKENDS_CPU_CODEGEN_EXECUTION_ENGINE_H_
