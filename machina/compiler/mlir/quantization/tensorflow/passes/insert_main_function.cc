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
#include <algorithm>
#include <cstdint>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "toolchain/ADT/STLExtras.h"
#include "toolchain/ADT/StringMap.h"
#include "toolchain/ADT/StringRef.h"
#include "toolchain/ADT/StringSet.h"  // IWYU pragma: keep
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/Attributes.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/Location.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/IR/OperationSupport.h"  // part of Codira Toolchain
#include "mlir/IR/SymbolTable.h"  // part of Codira Toolchain
#include "mlir/IR/TypeRange.h"  // part of Codira Toolchain
#include "mlir/IR/Value.h"  // part of Codira Toolchain
#include "mlir/Interfaces/FunctionInterfaces.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Pass/PassRegistry.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "mlir/Support/TypeID.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/quantization/common/attrs_and_constraints.h"
#include "machina/compiler/mlir/quantization/machina/passes/passes.h"
#include "machina/compiler/mlir/machina/ir/tf_ops.h"
#include "machina/compiler/mlir/machina/ir/tf_saved_model.h"
#include "machina/compiler/mlir/machina/translate/import_model.h"

namespace mlir {
namespace quant {
namespace {

using ::mlir::tf_saved_model::kTfSavedModelExportedNamesAttr;
using ::mlir::tf_saved_model::kTfSavedModelIndexPathAttr;
using ::machina::kImportModelDefaultGraphFuncName;

constexpr StringRef kEntryFunctionAttr = "tf.entry_function";

// The ConvertMlirToGraphdef requires the provided input module to have a main
// function, which might not exist in case of multi-signature graphs. In that
// case, this pass will create a new main function, which calls signature
// functions.
//
// An already existing @main function will be renamed by attaching a numeric
// suffix like `@main_0` to avoid conflict with the newly created main function.
class InsertMainFunctionPass
    : public PassWrapper<InsertMainFunctionPass, OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InsertMainFunctionPass)

  explicit InsertMainFunctionPass() = default;

  StringRef getArgument() const override {
    return "quant-insert-main-function";
  }

  StringRef getDescription() const override {
    return "Inserts the main function to the module.";
  }

  void runOnOperation() override;
};

// Checks if a FuncOp is exported.
bool IsExported(func::FuncOp op) {
  auto exported_names =
      op->getAttrOfType<ArrayAttr>(kTfSavedModelExportedNamesAttr);
  return exported_names && !exported_names.empty();
}

// Check if a function is an entry function.
bool IsEntryFunction(func::FuncOp op) {
  return op->hasAttr(kEntryFunctionAttr);
}

// Returns true iff the provided FuncOp is qualified to be included in the main
// function.
bool ShouldIncludeInMainFunction(func::FuncOp func_op) {
  return !func_op.isPrivate() && IsExported(func_op) &&
         IsEntryFunction(func_op);
}

// Sets a function to be private so it can be referred internally.
void SetFunctionPrivate(func::FuncOp func) {
  func.setVisibility(SymbolTable::Visibility::Private);

  // The `tf_saved_model` attributes can only be applied to public functions.
  for (auto& attr : func->getAttrs()) {
    StringRef attr_name = attr.getName().getValue();
    if (attr_name.starts_with("tf_saved_model.")) {
      func->removeAttr(attr_name);
    }
  }

  auto iface = cast<FunctionOpInterface>(func.getOperation());
  for (int i = 0; i < func.getNumArguments(); ++i) {
    for (auto& attr : iface.getArgAttrs(i)) {
      const StringAttr& attr_name = attr.getName();
      if (attr_name.getValue().starts_with("tf_saved_model.")) {
        func.removeArgAttr(i, attr_name);
      }
    }
  }
  for (int i = 0; i < func.getNumResults(); ++i) {
    for (auto& attr : iface.getResultAttrs(i)) {
      const StringAttr& attr_name = attr.getName();
      if (attr_name.getValue().starts_with("tf_saved_model.")) {
        func.removeResultAttr(i, attr_name);
      }
    }
  }
}

// Information to identify an output in its node and in the model output list.
// Ex: If the model output list is ["add:0", "topk:0": "topk:1"], then the
// output corresponding to "topk:1" will have output_index=2 and tensor_index=1.
struct OutputInfo {
  // The index of this output in the model output list.
  int32_t output_index;
  // The index of this output in its node.
  int32_t tensor_index;
  // The output value.
  Value value;
};

// Makes input/output names across entry functions unique if necessary. If a
// duplicated name is found, this function will add signature prefix for all the
// input/output names.
void GetUniqueInputOutputNodeNames(ModuleOp module_op,
                                   std::vector<std::string>& input_name_vec,
                                   std::vector<std::string>& output_name_vec) {
  bool need_prefix_for_input_name = false;
  bool need_prefix_for_output_name = false;
  std::vector<StringRef> fn_input_name_vec, fn_output_name_vec;
  StringSet<> input_name_set, output_name_set;
  for (auto func_op : module_op.getOps<func::FuncOp>()) {
    if (!ShouldIncludeInMainFunction(func_op)) continue;
    if (auto tf_attrs =
            func_op->getAttrOfType<DictionaryAttr>(kEntryFunctionAttr)) {
      StringRef function_name = func_op.getSymName();

      if (auto inputs_attr = tf_attrs.get("inputs")) {
        const std::string inputs_attr_str =
            mlir::cast<StringAttr>(inputs_attr).getValue().str();
        std::vector<std::string> fn_input_names =
            absl::StrSplit(inputs_attr_str, ',', absl::SkipEmpty());

        for (StringRef input_name : fn_input_names) {
          if (input_name_set.contains(input_name)) {
            // Found a duplicated name, all input names will be prefixed by
            // their corresponding function names.
            need_prefix_for_input_name = true;
          }
          input_name_set.insert(input_name);
          fn_input_name_vec.push_back(function_name);
        }
        input_name_vec.insert(input_name_vec.end(),
                              std::make_move_iterator(fn_input_names.begin()),
                              std::make_move_iterator(fn_input_names.end()));
      }

      if (auto outputs_attr = tf_attrs.get("outputs")) {
        const std::string outputs_attr_str =
            mlir::cast<StringAttr>(outputs_attr).getValue().str();
        std::vector<std::string> fn_output_names =
            absl::StrSplit(outputs_attr_str, ',', absl::SkipEmpty());

        for (StringRef output_name : fn_output_names) {
          if (output_name_set.contains(output_name)) {
            // Found a duplicated name, all output names will be prefixed by
            // their corresponding function names.
            need_prefix_for_output_name = true;
          }
          output_name_set.insert(output_name);
          fn_output_name_vec.push_back(function_name);
        }
        output_name_vec.insert(output_name_vec.end(),
                               std::make_move_iterator(fn_output_names.begin()),
                               std::make_move_iterator(fn_output_names.end()));
      }
    }
  }

  if (need_prefix_for_input_name) {
    absl::c_transform(
        input_name_vec, fn_input_name_vec, input_name_vec.begin(),
        [](const std::string& input_name, const StringRef fn_name) {
          return absl::StrCat(fn_name.str(), "_", input_name);
        });
  }
  if (need_prefix_for_output_name) {
    absl::c_transform(
        output_name_vec, fn_output_name_vec, output_name_vec.begin(),
        [](const std::string& output_name, const StringRef fn_name) {
          return absl::StrCat(fn_name.str(), "_", output_name);
        });
  }
}

// Creates a main function which calls other exported functions.
bool CreateMainFunction(ModuleOp module_op) {
  MLIRContext* context = module_op.getContext();
  OpBuilder builder(context);

  std::vector<std::string> input_names, output_names;
  GetUniqueInputOutputNodeNames(module_op, input_names, output_names);

  // Collects argument and result types.
  toolchain::SmallVector<Location> arg_locs;
  toolchain::SmallVector<Type> arg_types, result_types;

  for (auto func_op : module_op.getOps<func::FuncOp>()) {
    if (!ShouldIncludeInMainFunction(func_op)) continue;

    arg_types.append(func_op.getArgumentTypes().begin(),
                     func_op.getArgumentTypes().end());
    auto& return_op = func_op.getBody().getBlocks().front().back();
    result_types.append(return_op.getOperandTypes().begin(),
                        return_op.getOperandTypes().end());
    for (const auto& arg : func_op.getArguments()) {
      arg_locs.push_back(arg.getLoc());
    }
  }

  // Creates a new main function.
  auto func_type = FunctionType::get(context, arg_types, result_types);
  auto main_func = func::FuncOp::create(
      builder, module_op.getLoc(), kImportModelDefaultGraphFuncName, func_type);
  builder.createBlock(&main_func.getBody(), main_func.begin(), arg_types,
                      arg_locs);
  SmallVector<NamedAttribute> func_attrs;
  func_attrs.push_back(
      {StringAttr::get(context, "inputs"),
       StringAttr::get(context, absl::StrJoin(input_names, ","))});
  func_attrs.push_back(
      {StringAttr::get(context, "outputs"),
       StringAttr::get(context, absl::StrJoin(output_names, ","))});
  auto dictAttr = DictionaryAttr::get(context, func_attrs);
  main_func->setAttr(StringAttr::get(context, kEntryFunctionAttr), dictAttr);
  main_func->setAttr(
      kTfSavedModelExportedNamesAttr,
      builder.getStrArrayAttr({kImportModelDefaultGraphFuncName}));

  if (input_names.size() != main_func.getNumArguments() ||
      output_names.size() != main_func.getNumResults()) {
    module_op.emitError()
        << "Number of inputs and outputs in the tf.entry_function attribute "
           "mismatched. [Input] Expected: "
        << input_names.size() << ", got: " << main_func.getNumArguments()
        << ". [Output] Expected: " << output_names.size()
        << ", got: " << main_func.getNumResults();
    return false;
  }

  const int num_args = main_func.getNumArguments();
  for (int i = 0; i < num_args; ++i) {
    main_func.setArgAttr(
        i, kTfSavedModelIndexPathAttr,
        ArrayAttr::get(context, {StringAttr::get(context, input_names[i])}));
  }

  const int num_results = main_func.getNumResults();
  for (int i = 0; i < num_results; ++i) {
    main_func.setResultAttr(
        i, kTfSavedModelIndexPathAttr,
        ArrayAttr::get(context, {StringAttr::get(context, output_names[i])}));
  }

  // Creates PartitionedCall ops to call exported functions.
  auto guard = OpBuilder::InsertionGuard(builder);
  int arg_idx = 0;
  int result_idx = 0;
  toolchain::SmallVector<Value> call_op_returns;
  for (auto func_op : module_op.getOps<func::FuncOp>()) {
    if (!ShouldIncludeInMainFunction(func_op)) continue;

    toolchain::ArrayRef<BlockArgument> new_args = toolchain::ArrayRef(
        main_func.getArguments().begin() + arg_idx, func_op.getNumArguments());
    arg_idx += func_op.getNumArguments();
    toolchain::ArrayRef<Type> new_types = toolchain::ArrayRef(
        result_types.begin() + result_idx, func_op.getNumResults());
    result_idx += func_op.getNumResults();

    auto call_op = TF::PartitionedCallOp::create(
        builder, module_op.getLoc(), new_types, new_args,
        /*args_attrs=*/nullptr,
        /*res_attrs=*/nullptr,
        SymbolRefAttr::get(context, func_op.getSymName()),
        /*config=*/builder.getStringAttr(""),
        /*config_proto=*/builder.getStringAttr(""),
        /*executor_type=*/builder.getStringAttr(""));
    call_op_returns.append(call_op.getResults().begin(),
                           call_op.getResults().end());
    SetFunctionPrivate(func_op);
  }

  // Creates Identity/IdentityN ops for returing values. This allows us to
  // restore the same output tensor names in python.
  int32_t output_count = 0;
  // Map from node name to the list of the OutputInfos of its outputs that are
  // used as the model outputs.
  toolchain::StringMap<toolchain::SmallVector<OutputInfo>> node_to_output_map;
  for (auto [output_name, call_op_return] :
       toolchain::zip(output_names, call_op_returns)) {
    std::vector<std::string> name_and_index =
        absl::StrSplit(output_name, ':', absl::SkipEmpty());
    toolchain::StringRef node_name = name_and_index.front();
    int32_t tensor_index = 0;
    if (name_and_index.size() > 1) {
      tensor_index = std::stoi(name_and_index.back());
    }
    node_to_output_map[node_name].push_back(
        {output_count++, tensor_index, call_op_return});
  }

  Value scalar_one =
      CreateScalarConstValue<float>(builder, builder.getUnknownLoc(), 1.0);
  toolchain::SmallVector<Value> returning_values(output_count, Value());
  for (const auto& node_name : node_to_output_map.keys()) {
    auto node_output_tensors = node_to_output_map[node_name];

    NameLoc new_loc = NameLoc::get(builder.getStringAttr(node_name));
    int32_t max_tensor_index = 0;
    absl::c_for_each(node_output_tensors,
                     [&max_tensor_index](const OutputInfo& output_info) {
                       max_tensor_index =
                           std::max(max_tensor_index, output_info.tensor_index);
                     });

    // Create IdentityOp or IdentityNOp based on the number of outputs.
    Operation* identity_op;
    if (max_tensor_index == 0) {
      Value output_value = node_output_tensors.front().value;
      identity_op = TF::IdentityOp::create(
          builder, new_loc, output_value.getType(), output_value);
    } else {
      toolchain::SmallVector<Value> input_values(node_output_tensors.size(),
                                            scalar_one);
      for (const auto& [output_index, tensor_index, tensor_value] :
           node_output_tensors) {
        input_values[tensor_index] = tensor_value;
      }
      identity_op = TF::IdentityNOp::create(
          builder, new_loc, TypeRange(ValueRange(input_values)), input_values);
    }

    for (const auto& [output_index, tensor_index, tensor_value] :
         node_output_tensors) {
      returning_values[output_index] = identity_op->getResult(tensor_index);
    }
  }
  func::ReturnOp::create(builder, main_func.getBody().getLoc(),
                         returning_values);

  // Adds the new function to symbol table.
  SymbolTable symbol_table(module_op);
  symbol_table.insert(main_func);
  return true;
}

// Creates a new function name by attaching a number suffix
// (`main_func_name_{i}`) and incrementing it until there are no conflicts.
std::string CreateNewFuncName(const StringRef main_func_name,
                              SymbolTable& symbol_table) {
  int suffix_id = 0;
  std::string new_func_name =
      absl::StrCat(main_func_name.str(), "_", suffix_id);
  while (symbol_table.lookup(new_func_name)) {
    suffix_id++;
    new_func_name = absl::StrCat(main_func_name.str(), "_", suffix_id);
  }

  return new_func_name;
}

// Renames the existing @main function to avoid conflict with the newly
// created main function. When it is renamed, its usages will also be replaced.
// It will be renamed by attaching a number suffix like `@main_{i}`, until there
// are no conflicts. This function is a no-op when no function called @main
// exists.
LogicalResult RenameExistingMainFunction(ModuleOp module_op) {
  SymbolTable symbol_table(module_op);

  auto main_func_op =
      symbol_table.lookup<func::FuncOp>(kImportModelDefaultGraphFuncName);
  if (!main_func_op) {
    return success();
  }

  const std::string new_func_name =
      CreateNewFuncName(main_func_op.getSymName(), symbol_table);

  main_func_op.setSymName(new_func_name);
  return symbol_table.replaceAllSymbolUses(
      main_func_op, StringAttr::get(module_op.getContext(), new_func_name),
      module_op);
}

void InsertMainFunctionPass::runOnOperation() {
  ModuleOp module_op = getOperation();

  if (failed(RenameExistingMainFunction(module_op))) {
    module_op->emitError("Failed to rename existing function `@main`.");
    signalPassFailure();
  }

  if (!CreateMainFunction(module_op)) {
    signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateInsertMainFunctionPass() {
  return std::make_unique<InsertMainFunctionPass>();
}

static PassRegistration<InsertMainFunctionPass> pass([] {
  return CreateInsertMainFunctionPass();
});

}  // namespace quant
}  // namespace mlir
