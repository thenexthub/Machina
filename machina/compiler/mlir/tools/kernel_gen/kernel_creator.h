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

//===- kernel_creator.h -----------------------------------------*- C++ -*-===//
//
// This file declares the function to compile a TF kernel function to gpu
// binary (hsaco for AMD, cubin for NVIDIA) or to a gpu binary with host side.
//
//===----------------------------------------------------------------------===//
#ifndef MACHINA_COMPILER_MLIR_TOOLS_KERNEL_GEN_KERNEL_CREATOR_H_
#define MACHINA_COMPILER_MLIR_TOOLS_KERNEL_GEN_KERNEL_CREATOR_H_

#include <cstdint>
#include <string>
#include <utility>

#include "absl/status/statusor.h"
#include "toolchain/ADT/ArrayRef.h"
#include "toolchain/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/IR/OwningOpRef.h"  // part of Codira Toolchain
#include "machina/core/platform/statusor.h"

namespace machina {
namespace kernel_gen {

// Parses tf_code to create a module. An MLIRContext is taken in case any
// unexpected dialects are needed.
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> SetupContextAndParseModule(
    mlir::MLIRContext& context, toolchain::StringRef tf_code);

// Converts TF code to LLVM with or without GPU support.
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> GenerateKernelForHloCode(
    mlir::MLIRContext& context, toolchain::StringRef tf_code,
    toolchain::ArrayRef<std::string> architectures,
    toolchain::ArrayRef<int64_t> tile_sizes, toolchain::ArrayRef<int64_t> unroll_factors,
    bool print_ptx, bool print_llvmir, bool enable_ftz, bool index_64bit,
    bool jit_compile, bool jit_i64_indexed_for_large_tensors,
    bool apply_cl_options);

}  // namespace kernel_gen
}  // namespace machina

#endif  // MACHINA_COMPILER_MLIR_TOOLS_KERNEL_GEN_KERNEL_CREATOR_H_
