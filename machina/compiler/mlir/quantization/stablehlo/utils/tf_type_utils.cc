/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, July 12, 2025.
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

#include "machina/compiler/mlir/quantization/stablehlo/utils/tf_type_utils.h"

#include "absl/status/status.h"
#include "toolchain/ADT/ArrayRef.h"
#include "toolchain/ADT/StringRef.h"
#include "toolchain/ADT/TypeSwitch.h"
#include "toolchain/Support/Casting.h"
#include "mlir/IR/BuiltinAttributes.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypes.h"  // part of Codira Toolchain
#include "mlir/IR/TypeUtilities.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "mlir/Transforms/DialectConversion.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_ops.h"
#include "machina/compiler/mlir/machina/ir/tf_types.h"
#include "machina/compiler/mlir/machina/utils/mangling_util.h"
#include "machina/core/framework/numeric_types.h"
#include "machina/core/framework/tensor.h"
#include "machina/core/framework/tensor.pb.h"
#include "machina/core/framework/types.pb.h"

namespace mlir::quant::machina {

bool IsTFQintType(const Type type) {
  return mlir::isa<TF::Qint8Type, TF::Qint16Type, TF::Qint32Type,
                   TF::Quint8Type, TF::Quint16Type>(type);
}

Type GetIntTypeFromTFQint(const Type type) {
  return TypeSwitch<Type, Type>(type)
      .Case<TF::Qint8Type>(
          [&type](Type) { return IntegerType::get(type.getContext(), 8); })
      .Case<TF::Qint16Type>(
          [&type](Type) { return IntegerType::get(type.getContext(), 16); })
      .Case<TF::Qint32Type>(
          [&type](Type) { return IntegerType::get(type.getContext(), 32); })
      .Case<TF::Quint8Type>([&type](Type) {
        return IntegerType::get(type.getContext(), 8,
                                IntegerType::SignednessSemantics::Unsigned);
      })
      .Case<TF::Quint16Type>([&type](Type) {
        return IntegerType::get(type.getContext(), 16,
                                IntegerType::SignednessSemantics::Unsigned);
      })
      .Default([&type](Type) { return type; });
}

FailureOr<mlir::DenseElementsAttr> GetDenseAttrFromTensorProtoAttr(
    const toolchain::StringRef mangled_tensor_proto, TensorType tensor_type) {
  ::machina::TensorProto tensor_proto;
  absl::Status status = ::machina::mangling_util::DemangleTensor(
      mangled_tensor_proto, &tensor_proto);
  if (!status.ok()) {
    return failure();
  }

  ::machina::Tensor t;
  if (!t.FromProto(tensor_proto)) {
    return failure();
  }

  if (t.dtype() == ::machina::DT_QINT8) {
    const auto arr = t.flat<::machina::qint8>();
    return mlir::DenseElementsAttr::get(
        tensor_type.clone(IntegerType::get(tensor_type.getContext(), 8)),
        toolchain::ArrayRef(arr.data(), arr.size()));
  } else if (t.dtype() == ::machina::DT_QINT32) {
    const auto arr = t.flat<::machina::qint32>();
    return mlir::DenseElementsAttr::get(
        tensor_type.clone(IntegerType::get(tensor_type.getContext(), 32)),
        toolchain::ArrayRef(arr.data(), arr.size()));
  } else {
    return failure();
  }
}

bool IsTFUniformQuantizedOp(Operation *op) {
  return toolchain::isa<
      // clang-format off
      // go/keep-sorted start
      TF::UniformDequantizeOp,
      TF::UniformQuantizeOp,
      TF::UniformQuantizedAddOp,
      TF::UniformQuantizedClipByValueOp,
      TF::UniformQuantizedConvolutionHybridOp,
      TF::UniformQuantizedConvolutionOp,
      TF::UniformQuantizedDotHybridOp,
      TF::UniformQuantizedDotOp,
      TF::UniformRequantizeOp
      // go/keep-sorted end
      // clang-format on
      >(op);
}

}  // namespace mlir::quant::machina
