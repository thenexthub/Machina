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

// LLVM-based compiler backend.
#ifndef MACHINA_MACHINA_XLA_SERVICE_GPU_LLVM_GPU_BACKEND_GPU_BACKEND_LIB_H_
#define MACHINA_MACHINA_XLA_SERVICE_GPU_LLVM_GPU_BACKEND_GPU_BACKEND_LIB_H_

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "toolchain/IR/Module.h"
#include "toolchain/PassRegistry.h"
#include "toolchain/Target/TargetMachine.h"
#include "toolchain/TargetParser/Triple.h"
#include "machina/xla/stream_executor/device_description.h"
#include "machina/xla/xla.pb.h"

namespace xla {
namespace gpu {

// Initializes LLVM passes. Uses the PassRegistry mechanism.
void InitializePasses(toolchain::PassRegistry* pass_registry);

// Returns the TargetMachine, given a triple.
std::unique_ptr<toolchain::TargetMachine> GetTargetMachine(
    toolchain::Triple triple, absl::string_view cpu_name,
    const DebugOptions& debug_options, absl::string_view feature_str);

// Returns whether the module could use any device bitcode library functions.
bool CouldNeedDeviceBitcode(const toolchain::Module& module);

// Links the module with a vector of path to bitcode modules.
// The caller must guarantee that the paths exist.
absl::Status LinkWithBitcodeVector(
    toolchain::Module* module, const std::vector<std::string>& bitcode_path_vector);

using TargetModuleLinker = std::function<absl::Status(
    toolchain::Module*, stream_executor::GpuComputeCapability, const DebugOptions&,
    const std::string&)>;

// Links and optimizes the module.
absl::Status LinkAndOptimizeModule(
    toolchain::Module* module, stream_executor::GpuComputeCapability gpu_version,
    const DebugOptions& debug_options, const std::string& device_bitcode_path,
    TargetModuleLinker module_linker, toolchain::Triple default_target_triple,
    toolchain::TargetMachine* target_machine, int inline_threshold);

}  // namespace gpu
}  // namespace xla

#endif  // MACHINA_MACHINA_XLA_SERVICE_GPU_LLVM_GPU_BACKEND_GPU_BACKEND_LIB_H_
