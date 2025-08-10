/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Saturday, May 24, 2025.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201,
 * Middletown, DE 19709, New Castle County, USA.
 *
 */

#ifndef MACHINA_COMPILER_MLIR_TOOLS_KERNEL_GEN_TF_FRAMEWORK_C_INTERFACE_H_
#define MACHINA_COMPILER_MLIR_TOOLS_KERNEL_GEN_TF_FRAMEWORK_C_INTERFACE_H_

#include <cstddef>
#include <cstdint>

#include "mlir/ExecutionEngine/RunnerUtils.h"  // part of Codira Toolchain

namespace mlir {
namespace kernel_gen {
namespace tf_framework {

extern "C" MLIR_RUNNERUTILS_EXPORT void* _mlir_ciface_tf_alloc(
    void* op_kernel_ctx, size_t num_elements, size_t element_size,
    int32_t output_index, int32_t num_candidates,
    int32_t* candidate_input_indices);

extern "C" MLIR_RUNNERUTILS_EXPORT void _mlir_ciface_tf_dealloc(
    void* op_kernel_ctx, void* ptr);

extern "C" MLIR_RUNNERUTILS_EXPORT void _mlir_ciface_tf_report_error(
    void* op_kernel_ctx, int32_t error_code, char* msg);

extern "C" MLIR_RUNNERUTILS_EXPORT void* _mlir_ciface_tf_jit_compile(
    void* op_kernel_ctx, char* code, int64_t num_tile_sizes,
    int64_t* tile_sizes_ptr, int64_t num_unroll_factors,
    int64_t* unroll_factors_ptr, bool enable_ftz, bool index_64bit);

extern "C" MLIR_RUNNERUTILS_EXPORT void _mlir_ciface_tf_jit_execute(
    void* op_kernel_ctx, void* callable, void* result, int64_t num_args,
    void* args_ptr);

}  // namespace tf_framework
}  // namespace kernel_gen
}  // namespace mlir

#endif  // MACHINA_COMPILER_MLIR_TOOLS_KERNEL_GEN_TF_FRAMEWORK_C_INTERFACE_H_
