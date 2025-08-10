/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, June 15, 2025.
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

#include <memory>

#include "toolchain/ADT/ArrayRef.h"
#include "toolchain/ADT/STLExtras.h"
#include "toolchain/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/Attributes.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypes.h"  // part of Codira Toolchain
#include "mlir/IR/Diagnostics.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/TypeUtilities.h"  // part of Codira Toolchain
#include "mlir/IR/Types.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_types.h"
#include "machina/compiler/mlir/machina/utils/convert_tensor.h"
#include "machina/dtensor/cc/constants.h"
#include "machina/dtensor/mlir/ir/tf_dtensor.h"
#include "machina/dtensor/mlir/value_utils.h"

namespace machina {
namespace dtensor {

namespace {
#define GEN_PASS_DEF_DTENSORANNOTATEGLOBALSHAPE
#include "machina/dtensor/mlir/dtensor_passes.h.inc"

// Sets `_global_shape` attributes to argument/return values of `function`.
void AnnotateFunctionArgRetvalGlobalShapes(mlir::func::FuncOp function,
                                           mlir::OpBuilder* builder) {
  for (const auto& argument_type_and_index :
       toolchain::enumerate(function.getArgumentTypes())) {
    const int index = argument_type_and_index.index();
    const auto& argument_type = argument_type_and_index.value();
    // Extract TensorType from element of resource type to allow setting proper
    // global shape of resource types.
    if (auto resource_type = mlir::dyn_cast<mlir::TF::ResourceType>(
            mlir::getElementTypeOrSelf(argument_type))) {
      auto subtype = resource_type.getSubtypes();
      if (subtype.size() == 1) {
        // subtype returns a Array of TensorType -- if it contains more than one
        // Tensor type, we give up extracting the single TensorType inside the
        // subtype.
        function.setArgAttr(index, kGlobalShapeDialectAttr,
                            ConvertTypeToTensorShapeAttr(subtype[0]));
      }
    } else {
      function.setArgAttr(index, kGlobalShapeDialectAttr,
                          ConvertTypeToTensorShapeAttr(argument_type));
    }
  }

  for (const auto& retval_type_and_index :
       toolchain::enumerate(function.getFunctionType().getResults())) {
    const int index = retval_type_and_index.index();
    const auto& retval_type = retval_type_and_index.value();
    function.setResultAttr(index, kGlobalShapeDialectAttr,
                           ConvertTypeToTensorShapeAttr(retval_type));
  }
}

// Sets `_global_shape` attribute of an `op` with array of ShapeAttr of
// `outputs.
void AnnotateOperationGlobalShape(mlir::Operation* op,
                                  mlir::OpBuilder* builder) {
  toolchain::SmallVector<mlir::Attribute, 4> op_global_shape;
  op_global_shape.reserve(op->getNumResults());

  for (const auto& result_type : op->getResultTypes())
    op_global_shape.emplace_back(ConvertTypeToTensorShapeAttr(result_type));

  if (auto layout_op = mlir::dyn_cast<mlir::TF::DTensorLayout>(op)) {
    // Shape of Resource type is incorrect when it is a variable.
    // The global shape is undefined in this case; and usually we are supposed
    // to propagate the value shape due to how resource variable layout is
    // currently represented in DTensor.
    if (!IsResourceType(op->getResult(0))) {
      layout_op.setGlobalShapeAttr(op_global_shape[0]);
    }
  } else {
    op->setAttr(kGlobalShape, builder->getArrayAttr(op_global_shape));
  }
}

// Pass that annotates function argument/return values and all operation with
// `_global_shape` attribute. This will be used during SPMD expansion to
// preserve original global shape of operations in graph after shape has been
// modified to local shape.
struct DTensorAnnotateGlobalShape
    : public impl::DTensorAnnotateGlobalShapeBase<DTensorAnnotateGlobalShape> {
  void runOnOperation() override {
    mlir::MLIRContext& context = getContext();
    mlir::OpBuilder builder(&context);

    auto module = getOperation();
    module.walk([&](mlir::func::FuncOp function) {
      if (function.empty()) return;

      auto* terminator = function.getBody().front().getTerminator();
      AnnotateFunctionArgRetvalGlobalShapes(function, &builder);
      function.getBody().walk([&](mlir::Operation* op) {
        if (op == terminator) return;

        AnnotateOperationGlobalShape(op, &builder);
      });
    });
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateDTensorAnnotateGlobalShape() {
  return std::make_unique<DTensorAnnotateGlobalShape>();
}

}  // namespace dtensor
}  // namespace machina
