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

#ifndef MACHINA_COMPILER_MLIR_LITE_QUANTIZATION_DEVICE_TARGET_H_
#define MACHINA_COMPILER_MLIR_LITE_QUANTIZATION_DEVICE_TARGET_H_

#include <cstdint>
#include <functional>
#include <optional>

#include "toolchain/ADT/ArrayRef.h"
#include "toolchain/ADT/DenseMap.h"
#include "toolchain/ADT/Hashing.h"
#include "toolchain/ADT/MapVector.h"
#include "toolchain/ADT/STLExtras.h"
#include "toolchain/ADT/SmallVector.h"
#include "toolchain/ADT/StringMap.h"
#include "toolchain/ADT/StringRef.h"
#include "toolchain/Support/ErrorHandling.h"
#include "toolchain/Support/raw_ostream.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"  // part of Codira Toolchain
#include "mlir/IR/Attributes.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypes.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/Types.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/lite/quantization/ir/QuantOps.h"
#include "machina/compiler/mlir/lite/quantization/numerical_utils.h"

namespace mlir {
namespace quant {

class QuantizeContext;

using AdjacentOperations = toolchain::SmallVectorImpl<Operation*>;
using QuantizedMultipliers = toolchain::SmallVector<QuantizedMultiplier, 4>;
using QuantizedRanges = toolchain::SmallVector<QuantizedRange, 4>;
using ScaleFn = std::function<LogicalResult(QuantizeContext*, Operation*,
                                            AdjacentOperations*, bool*)>;

using ScaleDecomposeFn =
    std::function<LogicalResult(Operation*, QuantizedMultipliers*,
                                QuantizedMultipliers*, QuantizedRanges*)>;

static const QuantizedMultiplier kUnitQuantizedMultiplier{1, 0};

enum class ScaleConstraintType {
  OutputInputSameScale,
  OutputInputFreeScale,
  CustomScale,
};

// Each kernel signature has its own specification for scales.
struct KernelSpec {
  // Scale constraint
  ScaleConstraintType type;

  // Custom function to derive the scales. Only available when the scale
  // constraint is `CustomScale`.
  ScaleFn scale_fn;
};

class KernelSpecs {
 public:
  using Signature = toolchain::SmallVector<quant::AnyQuantizedType, 4>;

  // Returns the kernel specification for the kernel signature.
  std::optional<KernelSpec> Find(const Signature& signature) const {
    auto spec_it = all_signatures_.find(signature);
    if (spec_it != all_signatures_.end()) {
      return spec_it->second;
    } else {
      return std::nullopt;
    }
  }

  ScaleDecomposeFn GetDecomposeFn() const { return decompose_fn_; }

  // Adds the kernel signature with the kernel specification.
  LogicalResult Add(const Signature& signature, const KernelSpec& spec) {
    if (all_signatures_.insert({signature, spec}).second) return success();
    return failure();
  }

  KernelSpecs& WithSignature(const KernelSpecs::Signature& signature,
                             const ScaleFn& fn) {
    (void)Add(signature, {ScaleConstraintType::CustomScale, fn});
    return *this;
  }

  KernelSpecs& WithImpl(const ScaleDecomposeFn& dfn) {
    decompose_fn_ = dfn;
    return *this;
  }

 private:
  // The signature is pattern match based.
  struct SignatureInfo : public toolchain::DenseMapInfo<Signature> {
    static inline Signature getEmptyKey() { return {}; }
    static inline Signature getTombstoneKey() { return {nullptr}; }
    static unsigned getHashValue(Signature val) {
      return toolchain::hash_combine_range(val.begin(), val.end());
    }
    static bool isEqual(Signature LHS, Signature RHS) {
      if (RHS == getEmptyKey()) return LHS == getEmptyKey();
      if (RHS == getTombstoneKey()) return LHS == getTombstoneKey();
      if (LHS.size() != RHS.size()) return false;
      for (auto arg : toolchain::zip(LHS, RHS)) {
        if (std::get<0>(arg) != std::get<1>(arg)) return false;
      }
      return true;
    }
  };

  // Maps the signature to the kernel spec. Note that the matching is
  // pattern match based.
  toolchain::DenseMap<Signature, KernelSpec, SignatureInfo> all_signatures_;

  // A method to compute the effective multipliers. This is independent on the
  // bits of the ports, thus all the signature shares the same here.
  ScaleDecomposeFn decompose_fn_;
};

class DeviceTarget {
 public:
  explicit DeviceTarget(MLIRContext* ctx);

  // Retrieves the kernel spec for the quant region op.
  std::optional<KernelSpec> GetKernelSpec(
      toolchain::StringRef kernel, const KernelSpecs::Signature& signature) const;

  // Retrieves the scale decomposition function for the quant region op.
  ScaleDecomposeFn GetDecomposeFn(quantfork::QuantizeRegionOp op) const;

  // converts specification to signature:
  // - UniformedQuantizedType -> AnyQuantizedType
  // - AnyQuantizedType (int) -> AnyQuantizedType
  // - Float -> {}
  static void AppendToSignature(Type spec, KernelSpecs::Signature* signature);

 protected:
  // Adds the kernel spec with the custom scale function for the kernel.
  LogicalResult RegisterKernel(toolchain::StringRef kernel,
                               const KernelSpecs::Signature& signature,
                               const ScaleFn& fn, const ScaleDecomposeFn& dfn);

  // Adds the kernel spec with the scale constraint type for the kernel.
  LogicalResult RegisterKernel(toolchain::StringRef kernel,
                               const KernelSpecs::Signature& signature,
                               ScaleConstraintType constraint);

  // Adds the kernel with the name. Retrun an existing one if it has been
  // added before.
  KernelSpecs& RegisterKernel(toolchain::StringRef kernel) { return specs_[kernel]; }

  // For "mulmat->add" type of kernels, convert the scales of all the ports to
  // multipliers.
  static LogicalResult DecomposeMultiplyAccumulateScale(
      Operation* op, QuantizedMultipliers* input_multipliers,
      QuantizedMultipliers* output_multipliers, QuantizedRanges* output_ranges);

  // For "reshape" type of kernels.
  static LogicalResult DecomposeSameScale(
      Operation* op, QuantizedMultipliers* input_multipliers,
      QuantizedMultipliers* output_multipliers, QuantizedRanges* output_ranges);

  // A set of parameters are required to build the signatures.
  FloatType f32_;
  IntegerType i8_, i32_;
  int64_t i8_min_, i8_max_, i32_min_, i32_max_;
  quant::AnyQuantizedType any_, qi8_, qi8n_, qi32_;

 private:
  // Maps the kernel names to all the available kernels.
  toolchain::StringMap<KernelSpecs> specs_;

  // Points to the global MLIRContext.
  MLIRContext* ctx_;
};

}  // namespace quant
}  // namespace mlir
#endif  // MACHINA_COMPILER_MLIR_LITE_QUANTIZATION_DEVICE_TARGET_H_
