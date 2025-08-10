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

#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "toolchain/ADT/DenseMap.h"
#include "toolchain/ADT/STLExtras.h"
#include "toolchain/ADT/SmallVector.h"
#include "toolchain/ADT/StringRef.h"
#include "toolchain/Support/Casting.h"
#include "mlir/IR/Attributes.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinAttributes.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypes.h"  // part of Codira Toolchain
#include "mlir/IR/Diagnostics.h"  // part of Codira Toolchain
#include "mlir/IR/OpDefinition.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/Types.h"  // part of Codira Toolchain
#include "mlir/IR/Visitors.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/host_runtime/tfrt_ops.h"
#include "machina/compiler/mlir/machina/ir/tf_ops.h"
#include "machina/compiler/mlir/machina/ir/tf_types.h"
#include "machina/compiler/mlir/tfrt/transforms/ifrt/ifrt_constants.h"
#include "machina/xla/service/computation_placer.h"
#include "machina/xla/xla_data.pb.h"
#include "machina/core/platform/protobuf.h"  // IWYU pragma: keep
#include "machina/core/protobuf/tpu/compile_metadata.pb.h"
#include "machina/core/tfrt/ifrt/ifrt_config.pb.h"

namespace machina {
namespace ifrt_serving {
namespace {

#define GEN_PASS_DEF_SINKVARIABLEASNAMEDARRAYPASS
#define GEN_PASS_DECL_SINKVARIABLEASNAMEDARRAYPASS
#include "machina/compiler/mlir/tfrt/transforms/ifrt/passes.h.inc"  // IWYU pragma: keep

class SinkVariableAsNamedArrayPass
    : public impl::SinkVariableAsNamedArrayPassBase<
          SinkVariableAsNamedArrayPass> {
 public:
  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::OpBuilder builder(&getContext());

    // Rewrite ReadVariableOp with IfrtLoadVariableOp
    toolchain::SmallDenseMap<mlir::TF::ReadVariableOp, mlir::TF::IfrtLoadVariableOp>
        read_to_load;

    mlir::WalkResult walk_result =
        module.walk([&](mlir::TF::ReadVariableOp read_variable_op) {
          // TODO(b/319045348): consider use resource alias analysis for
          // this.
          auto var_handle = GetDefiningOp<mlir::TF::VarHandleOp>(
              read_variable_op.getResource());

          if (!var_handle) {
            read_variable_op->emitError(
                "ReadVariableOp has no defining VarHandleOp.");
            return mlir::WalkResult::interrupt();
          }

          // Avoid lowering ReadVariableOp to IfrtLoadVariableOp if the
          // assignment AssignVariableOp happens at the same module because
          // IfrtLoadVariableOp assumes asynchronous assignment of the variable.
          for (auto var_handle_user : var_handle->getUsers()) {
            if (toolchain::isa<mlir::TF::AssignVariableOp>(var_handle_user)) {
              return mlir::WalkResult::advance();
            }
          }
          std::vector<mlir::Type> result_types;
          result_types.push_back(mlir::RankedTensorType::get(
              {}, builder.getType<mlir::TF::StringType>()));
          result_types.push_back(read_variable_op.getResult().getType());

          builder.setInsertionPointAfter(read_variable_op);
          auto load_variable_op = mlir::TF::IfrtLoadVariableOp::create(
              builder, read_variable_op->getLoc(), result_types,
              var_handle.getResult());
          read_to_load[read_variable_op] = load_variable_op;

          return mlir::WalkResult::advance();
        });

    if (walk_result.wasInterrupted()) {
      return signalPassFailure();
    }

    // Rewrite ifrt call: variable tensors are sunk as attribute.
    // The runtime guarantees the binding of corresponding loaded ifrt array
    // based on attributes.
    mlir::WalkResult ifrt_call_walk_result =
        module.walk([&](mlir::TF::IfrtCallOp call) {
          IfrtArgConfigList ifrt_call_argument_configs;

          if (mlir::failed(BuildIfrtCallArgumentConfig(
                  call, ifrt_call_argument_configs))) {
            return mlir::WalkResult::interrupt();
          }

          if (!call.getVariableArgIndicesAttr().empty()) {
            call->emitError()
                << "Expect empty " << call.getVariableArgIndicesAttrName().str()
                << " attributes, but got "
                << call.getVariableArgIndicesAttr().size() << " elements";
            return mlir::WalkResult::interrupt();
          }
          if (call->getOpOperands().size() !=
              ifrt_call_argument_configs.size()) {
            call->emitError()
                << "IfrtCallOp got " << call->getOpOperands().size()
                << " operands, but expects "
                << ifrt_call_argument_configs.size();
            return mlir::WalkResult::interrupt();
          }

          toolchain::SmallVector<int> variable_arg_indices;
          toolchain::SmallVector<mlir::Value> updated_args;

          for (const auto& [arg_idx, arg] :
               toolchain::enumerate(ifrt_call_argument_configs)) {
            if (arg.is_variable) {
              variable_arg_indices.push_back(arg_idx);
              // Variable use the key from IfrtLoadVariable.
              updated_args.push_back(
                  read_to_load[arg.read_variable_op].getArrayKey());
            } else {
              // non variable
              updated_args.push_back(call->getOperand(arg_idx));
            }
          }

          builder.setInsertionPointAfter(call);
          auto updated_ifrt_call = mlir::TF::IfrtCallOp::create(
              builder, call->getLoc(), call.getResultTypes(), updated_args);

          updated_ifrt_call->setAttrs(call->getAttrs());
          // Update variable_arg_indices attribute.
          updated_ifrt_call.setVariableArgIndicesAttr(
              builder.getI32ArrayAttr(variable_arg_indices));

          call.replaceAllUsesWith(updated_ifrt_call);
          call.erase();
          return mlir::WalkResult::advance();
        });

    if (ifrt_call_walk_result.wasInterrupted()) {
      return signalPassFailure();
    }

    // Remove all ReadVariableOp after replacing the CPU usage of
    // ReadVariableOp.
    for (auto& [read_variable_op, load_variable_op] : read_to_load) {
      if (!read_variable_op->use_empty()) {
        // This variable tensor is used by CPU host.
        load_variable_op.setUsedByHost(true);

        // Replace CPU use of ReadVariableOp
        read_variable_op.replaceAllUsesWith(load_variable_op.getTensorFuture());
      }
      read_variable_op.erase();
    }
  }

 private:
  struct IfrtArgConfig {
    bool is_variable;
    mlir::TF::ReadVariableOp read_variable_op;
  };
  using IfrtArgConfigList = toolchain::SmallVector<IfrtArgConfig>;

  // Build argument configuration map of a IfrtCallOp.
  mlir::LogicalResult BuildIfrtCallArgumentConfig(mlir::TF::IfrtCallOp call,
                                                  IfrtArgConfigList& args) {
    for (const auto& [arg_idx, input] : toolchain::enumerate(call->getOperands())) {
      // Assuming the nested function calls are inlined.
      if (auto read_variable_op =
              GetDefiningOp<mlir::TF::ReadVariableOp>(input)) {
        args.push_back(
            {.is_variable = true, .read_variable_op = read_variable_op});
      } else {
        args.push_back({.is_variable = false});
      }
    }

    return mlir::success();
  }

  template <typename OpT>
  OpT GetDefiningOp(const mlir::Value& value) {
    mlir::Operation* op = value.getDefiningOp();

    while (op && !toolchain::isa<OpT>(op)) {
      if (toolchain::isa<mlir::TF::IdentityOp>(op)) {
        op = op->getOperand(0).getDefiningOp();
      } else {
        return nullptr;
      }
    }

    if (op != nullptr) {
      return toolchain::dyn_cast<OpT>(op);
    } else {
      return nullptr;
    }
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateSinkVariableAsNamedArrayPass() {
  return std::make_unique<SinkVariableAsNamedArrayPass>();
}

}  // namespace ifrt_serving
}  // namespace machina
