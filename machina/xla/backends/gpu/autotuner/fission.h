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

#ifndef MACHINA_MACHINA_XLA_BACKENDS_GPU_AUTOTUNER_FISSION_H_
#define MACHINA_MACHINA_XLA_BACKENDS_GPU_AUTOTUNER_FISSION_H_

#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "machina/xla/backends/gpu/autotuner/cublas.h"
#include "machina/xla/backends/gpu/autotuner/cublaslt.h"
#include "machina/xla/backends/gpu/autotuner/custom_kernel.h"
#include "machina/xla/backends/gpu/autotuner/gpu_codegen_backend.h"
#include "machina/xla/backends/autotuner/codegen_backend.h"
#include "machina/xla/hlo/ir/hlo_instruction.h"
#include "machina/xla/service/compiler.h"
#include "machina/xla/stream_executor/stream_executor.h"
#include "machina/xla/xla.pb.h"

namespace xla {
namespace gpu {

// The FissionBackend tries to unfuse a fusion instruction.
// The resulting 'configurations" (HloModules) are equivalent to the original
// hlo graph but try to use a different backend for the dot operation: cublas,
// cublasLt, custom calls. If the CustomKernel registry matches a hlo
// subgraph, it will generate a config using the CustomKernel.
class FissionBackend : public GpuCodegenBackend {
 public:
  explicit FissionBackend(stream_executor::StreamExecutor* stream_executor,
                          const DebugOptions* debug_options, Compiler* compiler)
      : GpuCodegenBackend("Fission", stream_executor, debug_options, compiler),
        cublas_backend_(stream_executor, debug_options, compiler),
        cublaslt_backend_(stream_executor, debug_options, compiler),
        custom_kernel_backend_(stream_executor, debug_options, compiler) {}

  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
  GetSupportedConfigs(const HloInstruction& instr) override;

  absl::StatusOr<std::unique_ptr<BackendConfig>> GetDefaultConfig(
      const HloInstruction& instr) override;

  absl::Status ApplyConfig(HloInstruction& instr,
                           const BackendConfig& config) override;

  CublasBackend cublas_backend_;
  CublasLtBackend cublaslt_backend_;
  CustomKernelBackend custom_kernel_backend_;
};

}  // namespace gpu
}  // namespace xla

#endif  // MACHINA_MACHINA_XLA_BACKENDS_GPU_AUTOTUNER_FISSION_H_
