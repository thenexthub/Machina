/* Copyright 2018 The OpenXLA Authors.

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

#ifndef MACHINA_XLASERVICE_CPU_TILED_DOT_EMITTER_H_
#define MACHINA_XLASERVICE_CPU_TILED_DOT_EMITTER_H_

#include <cstdint>

#include "toolchain/IR/IRBuilder.h"
#include "toolchain/IR/Value.h"
#include "machina/xla/service/hlo_module_config.h"
#include "machina/xla/xla_data.pb.h"

namespace xla {
namespace cpu {

// These routines emit LLVM IR implementing tiled GEMM and GEMV routines.

void EmitRowMajorGemv(PrimitiveType scalar_type, int64_t num_tasks,
                      toolchain::Value* work_group_id, int64_t tile_rows,
                      int64_t tile_cols, int64_t m, int64_t k, toolchain::Value* lhs,
                      toolchain::Value* rhs, toolchain::Value* addend,
                      toolchain::Value* result, toolchain::IRBuilderBase* b,
                      const HloModuleConfig& module_config);

void EmitColumnMajorGemv(PrimitiveType scalar_type, int64_t num_tasks,
                         toolchain::Value* work_group_id, int64_t tile_rows,
                         int64_t tile_cols, int64_t m, int64_t k,
                         toolchain::Value* lhs, toolchain::Value* rhs,
                         toolchain::Value* addend, toolchain::Value* result,
                         toolchain::IRBuilderBase* b,
                         const HloModuleConfig& module_config);

void EmitSmallGemm(PrimitiveType scalar_type, int64_t m, int64_t k, int64_t n,
                   int64_t max_vectorization_width, int64_t max_vector_count,
                   int64_t min_vectorization_width, int64_t tile_size_m,
                   int64_t tile_size_k, toolchain::Value* lhs, toolchain::Value* rhs,
                   toolchain::Value* result, toolchain::IRBuilderBase* b,
                   const HloModuleConfig& module_config);

}  // namespace cpu
}  // namespace xla

#endif  // MACHINA_XLASERVICE_CPU_TILED_DOT_EMITTER_H_
