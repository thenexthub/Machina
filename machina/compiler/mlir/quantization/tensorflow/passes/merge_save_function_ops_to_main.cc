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
#include <memory>
#include <utility>

#include "absl/algorithm/container.h"
#include "toolchain/ADT/STLExtras.h"
#include "toolchain/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinAttributes.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypes.h"  // part of Codira Toolchain
#include "mlir/IR/IRMapping.h"  // part of Codira Toolchain
#include "mlir/IR/Location.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/IR/Value.h"  // part of Codira Toolchain
#include "mlir/IR/ValueRange.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Pass/PassRegistry.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "mlir/Support/TypeID.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/quantization/machina/passes/constants.h"
#include "machina/compiler/mlir/quantization/machina/passes/manipulate_model_attr.h"
#include "machina/compiler/mlir/quantization/machina/passes/passes.h"
#include "machina/compiler/mlir/machina/ir/tf_executor.h"
#include "machina/compiler/mlir/machina/ir/tf_ops.h"
#include "machina/compiler/mlir/machina/ir/tf_saved_model.h"
#include "machina/compiler/mlir/machina/ir/tf_types.h"
#include "machina/compiler/mlir/machina/translate/import_model.h"

namespace mlir {
namespace quant {
namespace {

using ::mlir::tf_executor::FetchOp;
using ::mlir::tf_executor::GraphOp;
using ::mlir::tf_executor::IslandOp;
using ::mlir::tf_saved_model::kTfSavedModelIndexPathAttr;
using ::machina::kImportModelDefaultGraphFuncName;

class MergeSaveFunctionOpsToMainPass
    : public PassWrapper<MergeSaveFunctionOpsToMainPass,
                         OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MergeSaveFunctionOpsToMainPass)

  explicit MergeSaveFunctionOpsToMainPass() = default;

  StringRef getArgument() const override {
    return "quant-merge-save-function-ops-to-main";
  }

  StringRef getDescription() const override {
    return "Merge the save function's ops to the main function. The save "
           "function will be removed after the pass.";
  }

  void runOnOperation() override;
};

// Returns true iff func_op has either no Region or the body has no Blocks.
bool IsFuncOpEmpty(func::FuncOp func_op) {
  return func_op->getNumRegions() == 0 || func_op.getBody().empty();
}

// Gets the GraphOp from the function op. Returns an empty op iff it doesn't
// exist.
GraphOp GetGraphOpFromFuncOp(func::FuncOp func_op) {
  if (IsFuncOpEmpty(func_op)) return {};

  auto graph_op_range = func_op.front().without_terminator();
  if (toolchain::hasSingleElement(graph_op_range)) {
    // The pass runs on a valid tf_executor dialect, so the op should be the
    // GraphOp.
    return cast<GraphOp>(graph_op_range.begin());
  }

  return {};
}

// Gets the "main" function from the module. Returns an empty op iff it doesn't
// exist.
func::FuncOp GetMainFunction(ModuleOp module_op) {
  const auto main_func_id =
      StringAttr::get(module_op.getContext(), kImportModelDefaultGraphFuncName);
  auto func_ops = module_op.getOps<func::FuncOp>();
  auto main_func_itr = absl::c_find_if(func_ops, [&main_func_id](auto func_op) {
    return func_op.getName() == main_func_id;
  });

  if (main_func_itr == func_ops.end()) return {};
  return *main_func_itr;
}

func::FuncOp GetSaveFuncOp(ModuleOp module_op) {
  for (auto func_op : module_op.getOps<func::FuncOp>()) {
    if (func_op.getSymName() == kTfQuantSaveFuncName) return func_op;
  }

  return nullptr;
}

// Adds the file prefix argument to `main_func_op`. The file prefix argument
// is the argument whose "tf_saved_model.index_path" attribute has
// "__tf_file_prefix". Its type is `tensor<!tf_type.string>`. Also, the value
// "__tf_file_prefix:0" is appended to the "tf.entry_function" attribute's
// "inputs" key.
BlockArgument CreateFilePrefixArg(func::FuncOp main_func_op) {
  Builder builder(main_func_op);

  // Add a new argument of type `tensor<!tf_type.string>` and update the
  // function type.
  auto file_prefix_arg_type =
      RankedTensorType::get(/*shape=*/{}, builder.getType<TF::StringType>());
  BlockArgument new_file_prefix_arg =
      main_func_op.getBody().front().addArgument(
          file_prefix_arg_type,
          NameLoc::get(builder.getStringAttr(kTfFilePrefix)));

  SmallVector<Type> input_types(main_func_op.getArgumentTypes());
  input_types.emplace_back(file_prefix_arg_type);

  main_func_op.setType(
      builder.getFunctionType(input_types, main_func_op.getResultTypes()));

  // Add "__tf_file_prefix" to the "tf_saved_model.index_path" attribute for the
  // newly created argument.
  main_func_op.setArgAttr(new_file_prefix_arg.getArgNumber(),
                          /*name=*/kTfSavedModelIndexPathAttr,
                          /*value=*/builder.getStrArrayAttr({kTfFilePrefix}));

  // Append the "__tf_file_prefix:0" to the "tf.entry_function" attribute's
  // item keyed by "inputs".
  AddEntryFunctionInput(Twine(kTfFilePrefix).concat(":0").str(), main_func_op);

  return new_file_prefix_arg;
}

// Finds the file prefix argument from `main_func_op`. The file prefix argument
// is the argument whose "tf_saved_model.index_path" attribute has
// "__tf_file_prefix". If such an argument doesn't exist, returns a null value.
BlockArgument GetFilePrefixArg(func::FuncOp main_func_op) {
  for (int i = 0; i < main_func_op.getNumArguments(); i++) {
    auto index_path_attr =
        main_func_op.getArgAttrOfType<ArrayAttr>(i, kTfSavedModelIndexPathAttr);
    if (index_path_attr && !index_path_attr.empty() &&
        mlir::cast<StringAttr>(index_path_attr[0]) == kTfFilePrefix) {
      return main_func_op.getArgument(i);
    }
  }
  return {};
}

// Returns the existing file prefix argument from the `main_func_op`. The file
// prefix argument is the argument whose "tf_saved_model.index_path" attribute
// has "__tf_file_prefix". If such an argument doesn't exist, creates a new file
// prefix argument and returns it.
BlockArgument GetOrCreateFilePrefixArg(func::FuncOp main_func_op) {
  if (BlockArgument main_file_prefix_arg = GetFilePrefixArg(main_func_op);
      main_file_prefix_arg) {
    return main_file_prefix_arg;
  } else {
    return CreateFilePrefixArg(main_func_op);
  }
}

// Clones ops from `src_graph_op` to `dst_graph_op`. The `dst_graph_op`'s
// `FetchOp` will be used without modified. Returns the fetch operands from the
// `scr_graph_op`.
Value CloneGraphOps(GraphOp src_graph_op, GraphOp dst_graph_op,
                    IRMapping& mapper) {
  Block& main_body = dst_graph_op.GetBody();

  // Take the reference of the main graph's FetchOp to later move to the end.
  FetchOp main_fetch_op = dst_graph_op.GetFetch();

  Block& save_func_body = src_graph_op.GetBody();
  for (Operation& op : save_func_body.without_terminator()) {
    main_body.push_back(op.clone(mapper));
  }

  // Relocate the main function's FetchOp to the last.
  main_body.push_back(main_fetch_op->clone(mapper));
  main_fetch_op.erase();

  auto cloned_fetch_op = cast<FetchOp>(src_graph_op.GetFetch()->clone(mapper));
  Value control_fetch = *cloned_fetch_op.getFetches().begin();
  cloned_fetch_op.erase();

  return control_fetch;
}

// Creates a new `IdentityOp` wrapped by an `IslandOp`. The identity op returns
// the `main_file_prefix_arg` and has control dependencies to `control_inputs`.
IslandOp CreateFilePrefixIdentityOp(const BlockArgument main_file_prefix_arg,
                                    const ArrayRef<Value> control_inputs,
                                    GraphOp main_graph_op) {
  MLIRContext& ctx = *main_graph_op.getContext();
  const auto name_loc = NameLoc::get(StringAttr::get(&ctx, kTfQuantSaveOpName));

  auto builder = OpBuilder::atBlockTerminator(&main_graph_op.GetBody());
  // Create an IslandOp that will wrap the IdentityOp. Add a control dependency
  // for the newly copied save function.
  auto wrapper_island_op = IslandOp::create(
      builder, name_loc, TypeRange{main_file_prefix_arg.getType()},
      tf_executor::ControlType::get(&ctx), ValueRange(control_inputs));
  wrapper_island_op.getBody().emplaceBlock();

  builder.setInsertionPointToStart(&wrapper_island_op.GetBody());
  auto identity_op = TF::IdentityOp::create(
      builder, name_loc, /*result_types=*/main_file_prefix_arg.getType(),
      /*input=*/main_file_prefix_arg);

  tf_executor::YieldOp::create(builder, name_loc, identity_op.getResult());

  return wrapper_island_op;
}

// Appends `value` to the arguments of the `FetchOp` of `graph_op`.
void AppendValueToFetch(GraphOp graph_op, Value value) {
  FetchOp old_main_fetch = graph_op.GetFetch();
  auto fetches = toolchain::to_vector(old_main_fetch.getFetches());
  fetches.emplace_back(value);

  auto builder = OpBuilder::atBlockTerminator(&graph_op.GetBody());
  FetchOp::create(builder, old_main_fetch.getLoc(), std::move(fetches));
  old_main_fetch.erase();
}

void MergeSaveFunctionOpsToMain(func::FuncOp save_func_op,
                                func::FuncOp main_func_op) {
  GraphOp main_graph_op = GetGraphOpFromFuncOp(main_func_op);
  if (!main_graph_op) return;

  GraphOp save_func_graph_op = GetGraphOpFromFuncOp(save_func_op);
  if (!save_func_graph_op) return;

  IRMapping mapper{};
  BlockArgument main_file_prefix_arg = GetOrCreateFilePrefixArg(main_func_op);
  // TODO(b/268452435): This part assumes that the save function is always valid
  // and has the argument. Add a validation function to filter out any invalid
  // inputs.
  mapper.map(save_func_op.getArgument(0), main_file_prefix_arg);

  Value save_control_fetch =
      CloneGraphOps(save_func_graph_op, main_graph_op, mapper);

  IslandOp file_prefix_identity_wrapper = CreateFilePrefixIdentityOp(
      main_file_prefix_arg, /*control_inputs=*/{save_control_fetch},
      main_graph_op);

  // Adds the newly created identity op's control output to the main's fetches.
  AppendValueToFetch(main_graph_op, file_prefix_identity_wrapper.getControl());
}

}  // namespace

void MergeSaveFunctionOpsToMainPass::runOnOperation() {
  ModuleOp module_op = getOperation();

  func::FuncOp main_func_op = GetMainFunction(module_op);
  if (!main_func_op) {
    module_op.emitError("Main function op not found.");
    return signalPassFailure();
  }

  func::FuncOp save_func_op = GetSaveFuncOp(module_op);
  if (!save_func_op) return;

  MergeSaveFunctionOpsToMain(save_func_op, main_func_op);

  // Erase the save function when all ops are successfully cloned.
  save_func_op.erase();
}

std::unique_ptr<OperationPass<ModuleOp>>
CreateMergeSaveFunctionOpsToMainPass() {
  return std::make_unique<MergeSaveFunctionOpsToMainPass>();
}

static PassRegistration<MergeSaveFunctionOpsToMainPass> pass([] {
  return CreateMergeSaveFunctionOpsToMainPass();
});

}  // namespace quant
}  // namespace mlir
