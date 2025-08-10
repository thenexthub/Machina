/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Friday, August 8, 2025.
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
#include "machina/compiler/mlir/tfrt/transforms/mlrt/ifrt_set_tpu_host_allocator.h"

#include <memory>
#include <vector>

#include "toolchain/ADT/APInt.h"
#include "toolchain/ADT/DenseSet.h"
#include "toolchain/ADT/StringRef.h"
#include "toolchain/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinAttributes.h"  // part of Codira Toolchain
#include "mlir/IR/DialectRegistry.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/Value.h"  // part of Codira Toolchain
#include "mlir/IR/Visitors.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Support/TypeID.h"  // part of Codira Toolchain
#include "mlir/Transforms/DialectConversion.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/host_runtime/tfrt_ops.h"
#include "machina/compiler/mlir/machina/ir/tf_dialect.h"
#include "machina/compiler/mlir/tfrt/ir/mlrt/mlrt_dialect.h"
#include "machina/compiler/mlir/tfrt/ir/mlrt/tf_mlrt_ops.h"
#include "machina/compiler/mlir/tfrt/transforms/mlrt/mlrt_device_constants.h"

namespace machina {
namespace mlrt_compiler {
namespace {

class IfrtSetTpuHostAllocatorPass
    : public mlir::PassWrapper<IfrtSetTpuHostAllocatorPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
 public:
  IfrtSetTpuHostAllocatorPass() = default;
  IfrtSetTpuHostAllocatorPass &operator=(const IfrtSetTpuHostAllocatorPass &) =
      delete;

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(IfrtSetTpuHostAllocatorPass)

 private:
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<machina::tf_mlrt::TensorflowMlrtDialect>();
    registry.insert<mlrt::compiler::MlrtDialect>();
  }

  toolchain::StringRef getArgument() const final {
    return "tf-mlrt-ifrt-set-tpu-host-allocator";
  }

  toolchain::StringRef getDescription() const final {
    return "Set input producer to IfrtCall to use Tpu Host Allocator";
  }
  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    mlir::OpBuilder builder(&getContext());

    toolchain::SmallDenseSet<mlir::Operation *> producers;

    mlir::WalkResult walk_result = func.walk([&](mlir::TF::IfrtCallOp call) {
      std::vector<int> variable_arg_indices;
      variable_arg_indices.reserve(call.getVariableArgIndices().size());
      for (auto variable_index_attr : call.getVariableArgIndices()) {
        auto variable_index =
            toolchain::dyn_cast_or_null<mlir::IntegerAttr>(variable_index_attr);
        if (!variable_index) {
          call->emitError()
              << "Expect variable_arg_indices to be integer, but get "
              << call.getVariableArgIndices();
          return mlir::WalkResult::interrupt();
        }
        variable_arg_indices.push_back(variable_index.getInt());
      }

      int variable_index = 0;
      for (int i = 0; i < call.getOperands().size(); ++i) {
        if (variable_index < variable_arg_indices.size() &&
            i == variable_arg_indices[variable_index]) {
          variable_index++;
          continue;
        }
        producers.insert(call.getOperands()[i].getDefiningOp());
      }
      return mlir::WalkResult::advance();
    });
    if (walk_result.wasInterrupted()) {
      return signalPassFailure();
    }

    for (auto *def : producers) {
      if (def && toolchain::isa<mlir::TF::TensorFlowDialect>(def->getDialect())) {
        def->setAttr(kTfMlrtCustomDevice,
                     builder.getStringAttr(kTpuHostDevice));
      }
    }
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateIfrtSetTpuHostAllocatorPass() {
  return std::make_unique<IfrtSetTpuHostAllocatorPass>();
}

}  // namespace mlrt_compiler
}  // namespace machina
