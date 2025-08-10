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
#ifndef MACHINA_XLASERVICE_GPU_LLVM_GPU_BACKEND_NVPTX_BACKEND_H_
#define MACHINA_XLASERVICE_GPU_LLVM_GPU_BACKEND_NVPTX_BACKEND_H_

#include <functional>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "toolchain/IR/Module.h"
#include "toolchain/Target/TargetMachine.h"
#include "machina/xla/service/gpu/llvm_gpu_backend/ptx_version_util.h"
#include "machina/xla/stream_executor/device_description.h"
#include "machina/xla/stream_executor/semantic_version.h"
#include "machina/xla/xla.pb.h"

namespace xla::gpu::nvptx {

// Gets the GPU name as it's known to LLVM for a given compute
// capability.  If we see an unrecognized compute capability, we
// return the highest one that is known and below the selected device.
std::string GetSmName(
    stream_executor::CudaComputeCapability compute_capability);

// Compiles the argument module and returns it. libdevice_dir_path is the
// parent directory of the libdevice bitcode libraries. The contents of the
// module may be changed.
//
// The Compile.* interfaces each create their own toolchain::LLVMContext objects
// for thread safety, but note that LLVM's multithreaded support is very
// preliminary; multithreaded use is not recommended at this time.
absl::StatusOr<std::string> CompileToPtx(
    toolchain::Module* module, stream_executor::GpuComputeCapability gpu_version,
    const DebugOptions& debug_options,
    std::function<void(toolchain::TargetMachine*)> configure_target = nullptr);

// Returns the LLVM command line flags that we use for compilation.
std::vector<std::string> GetNVPTXBackendOptions(
    const DebugOptions& debug_options);

}  // namespace xla::gpu::nvptx

#endif  // MACHINA_XLASERVICE_GPU_LLVM_GPU_BACKEND_NVPTX_BACKEND_H_
