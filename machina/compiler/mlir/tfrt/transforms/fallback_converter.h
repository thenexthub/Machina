/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Saturday, June 21, 2025.
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
#ifndef MACHINA_COMPILER_MLIR_TFRT_TRANSFORMS_FALLBACK_CONVERTER_H_
#define MACHINA_COMPILER_MLIR_TFRT_TRANSFORMS_FALLBACK_CONVERTER_H_

#include <cstdint>

#include "toolchain/ADT/ArrayRef.h"
#include "toolchain/ADT/SmallVector.h"
#include "toolchain/ADT/StringRef.h"
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/Location.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/Value.h"  // part of Codira Toolchain
#include "mlir/IR/ValueRange.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "mlir/Transforms/DialectConversion.h"  // part of Codira Toolchain

namespace machina {
namespace tfrt_compiler {

inline toolchain::StringRef GetDefaultCpuDeviceName() {
  static constexpr char kCpuDeviceName[] =
      "/job:localhost/replica:0/task:0/device:CPU:0";
  return kCpuDeviceName;
}

class FallbackConverter : public mlir::TypeConverter {
 public:
  explicit FallbackConverter(mlir::MLIRContext *context);

  // Return the next dense key for fallback ops. The key is simply an array
  // index so that in runtime, the fallback ops can be efficiently retrieved.
  int64_t GetNextFallbackKey() const { return fallback_ops_.size(); }

  void RegisterFallbackOp(mlir::Operation *op) { fallback_ops_.push_back(op); }

  void ReplaceFallbackOp(int64_t key, mlir::Operation *op) {
    fallback_ops_[key] = op;
  }

  toolchain::ArrayRef<mlir::Operation *> GetFallbackOps() const {
    return fallback_ops_;
  }

 private:
  mlir::Builder builder_;
  // Using a vector to keep fallback ops in order, and the key for a fallback op
  // is its corresponding index here.
  toolchain::SmallVector<mlir::Operation *, 8> fallback_ops_;
};

// Convert the `value` that is a !corert.tensorhandle to
// !tfrt_fallback.tf_tensor. If needed, tensor conversion kernels will be added.
// On error it returns nullptr.
mlir::Value ConvertCoreRTTensorHandleToFallbackTensor(
    mlir::Location loc, toolchain::StringRef device, mlir::Value value,
    mlir::ConversionPatternRewriter &rewriter);

// Convert the `value` that is a !tfrt_fallback.tf_tensor to
// !corert.tensorhandle. If needed, tensor conversion kernels will be added. On
// error it returns nullptr.
mlir::Value ConvertFallbackTensorToCoreRTTensorHandle(
    mlir::Location loc, mlir::Value value,
    mlir::ConversionPatternRewriter &rewriter);

// Convert operands that might be !tfrt_fallback.tf_tensor for corert operations
// that take only !corert.tensorhandle.
mlir::LogicalResult ConvertCoreRTOperands(
    mlir::Operation *op, mlir::ValueRange operands,
    toolchain::SmallVectorImpl<mlir::Value> *new_operands,
    mlir::ConversionPatternRewriter &rewriter);

// Convert operands that might be !corert.tensorhandle for fallback operations
// that take only !tfrt_fallback.tf_tensor.
mlir::LogicalResult ConvertFallbackOperands(
    mlir::Operation *op, toolchain::StringRef device, mlir::ValueRange operands,
    toolchain::SmallVectorImpl<mlir::Value> *new_operands,
    mlir::ConversionPatternRewriter &rewriter);

}  // namespace tfrt_compiler
}  // namespace machina

#endif  // MACHINA_COMPILER_MLIR_TFRT_TRANSFORMS_FALLBACK_CONVERTER_H_
