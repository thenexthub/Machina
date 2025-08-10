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

// The full pipeline of converting jax random include 2 steps.
// 1. Rename the jax random functions to tflite wrapped functions with the aid
//    of "jax.named_call". For example, in the dumped hlo, the
//    jax.random.uniform will have name "tfl_wrapped_jax_random_uniform".
// 2. Replace the body of "tfl_wrapped_jax_random_uniform" and
//    "tfl_wrapped_jax_random_normal" with tfl.CustomOp("RandomUniform") and
//     tfl.CustomOp("RandomStandardNormal"), respectively.

#include <string>

#include "toolchain/ADT/ArrayRef.h"
#include "toolchain/ADT/STLExtras.h"
#include "toolchain/ADT/SmallVector.h"
#include "toolchain/ADT/StringRef.h"
#include "toolchain/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/Dialect/Quant/IR/Quant.h"  // part of Codira Toolchain
#include "mlir/IR/Attributes.h"  // part of Codira Toolchain
#include "mlir/IR/Block.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinAttributes.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypes.h"  // part of Codira Toolchain
#include "mlir/IR/ImplicitLocOpBuilder.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/IR/OperationSupport.h"  // part of Codira Toolchain
#include "mlir/IR/PatternMatch.h"  // part of Codira Toolchain
#include "mlir/IR/Region.h"  // part of Codira Toolchain
#include "mlir/IR/TypeRange.h"  // part of Codira Toolchain
#include "mlir/IR/Visitors.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "mlir/Transforms/DialectConversion.h"  // part of Codira Toolchain
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "machina/compiler/mlir/lite/ir/tfl_ops.h"
#include "machina/compiler/mlir/machina/ir/tf_dialect.h"
#include "machina/compiler/mlir/machina/ir/tf_ops.h"

namespace mlir {
namespace TFL {
namespace {
#define GEN_PASS_DEF_LEGALIZEJAXRANDOMPASS
#include "machina/compiler/mlir/lite/transforms/passes.h.inc"

struct LegalizeJaxRandomPass
    : public impl::LegalizeJaxRandomPassBase<LegalizeJaxRandomPass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LegalizeJaxRandomPass)

  void runOnOperation() override;
};

inline ConstBytesAttr CustomOption(ImplicitLocOpBuilder *builder,
                                   const std::string &content) {
  return ConstBytesAttr::get(builder->getContext(),
                             StringRef(content.data(), content.size()));
}

inline bool IsJaxRandomUniform(mlir::func::FuncOp func) {
  return func.getName().contains("tfl_wrapped_jax_random_uniform");
}

inline bool IsJaxRandomNormal(mlir::func::FuncOp func) {
  return func.getName().contains("tfl_wrapped_jax_random_normal");
}

void LegalizeJaxRandomPass::runOnOperation() {
  auto func = getOperation();
  if (!IsJaxRandomUniform(func) && !IsJaxRandomNormal(func)) return;
  auto result_tuple_ty =
      mlir::dyn_cast_or_null<TupleType>(func.getFunctionType().getResult(0));
  if (!result_tuple_ty) return;
  if (result_tuple_ty.size() != 1) return;
  auto result_ty = mlir::dyn_cast<ShapedType>(result_tuple_ty.getType(0));

  func.eraseBody();
  func.addEntryBlock();
  ImplicitLocOpBuilder builder(func.getLoc(), func.getBody());
  toolchain::SmallVector<int32_t> result_shape_i32;
  auto result_shape = result_ty.getShape();
  for (auto element : result_shape) {
    result_shape_i32.push_back(static_cast<int32_t>(element));
  }
  auto result_shape_attr = builder.getI32TensorAttr(result_shape_i32);
  Value result_shape_tensor =
      builder.create<stablehlo::ConstantOp>(result_shape_attr);
  auto custom_code =
      IsJaxRandomUniform(func) ? "RandomUniform" : "RandomStandardNormal";

  toolchain::SmallVector<Type> result_ty_vec({result_ty});
  toolchain::SmallVector<Value> result_shape_tensor_vec({result_shape_tensor});
  auto attr = CustomOption(&builder, "");
  Value random_result =
      builder
          .create<TFL::CustomOp>(TypeRange(result_ty_vec),
                                 ValueRange(result_shape_tensor_vec),
                                 custom_code, attr)
          .getResult(0);
  Value tulple_result = builder.create<stablehlo::TupleOp>(random_result);
  builder.create<mlir::func::ReturnOp>(tulple_result);
}
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> CreateLegalizeJaxRandomPass() {
  return std::make_unique<LegalizeJaxRandomPass>();
}

}  // namespace TFL
}  // namespace mlir
