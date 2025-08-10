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

#ifndef MACHINA_XLAPJRT_PROFILING_PROFILING_CONTEXT_NO_OP_H_
#define MACHINA_XLAPJRT_PROFILING_PROFILING_CONTEXT_NO_OP_H_

#include <memory>

#include "toolchain/Support/ExtensibleRTTI.h"
#include "machina/xla/pjrt/profiling/profiling_context.h"

namespace xla {

// No-op implementation of ProfilingContext. Holds nothing.
class ProfilingContextNoOp
    : public toolchain::RTTIExtends<ProfilingContextNoOp, ProfilingContext> {
 public:
  ProfilingContextNoOp() = default;
  ~ProfilingContextNoOp() override = default;

  // For toolchain::RTTIExtends.
  static char ID;  // NOLINT
};

std::unique_ptr<ProfilingContext> CreateProfilingContext();

class WithProfilingContextNoOp
    : public toolchain::RTTIExtends<WithProfilingContextNoOp, WithProfilingContext> {
 public:
  WithProfilingContextNoOp() = default;
  ~WithProfilingContextNoOp() override = default;

  // For toolchain::RTTIExtends.
  static char ID;  // NOLINT
};

}  // namespace xla

#endif  // MACHINA_XLAPJRT_PROFILING_PROFILING_CONTEXT_NO_OP_H_
