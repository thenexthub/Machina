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
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "toolchain/ADT/ArrayRef.h"
#include "toolchain/ADT/STLExtras.h"
#include "toolchain/ADT/SmallVector.h"
#include "toolchain/ADT/StringRef.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"  // part of Codira Toolchain
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/Dialect/Quant/IR/Quant.h"  // part of Codira Toolchain  // IWYU pragma: keep
#include "mlir/IR/Attributes.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinAttributes.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/IR/OwningOpRef.h"  // part of Codira Toolchain
#include "mlir/IR/SymbolTable.h"  // part of Codira Toolchain
#include "mlir/IR/Visitors.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "stablehlo/dialect/ChloOps.h"  // from @stablehlo  // IWYU pragma: keep
#include "stablehlo/dialect/Serialization.h"  // from @stablehlo
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo  // IWYU pragma: keep
#include "stablehlo/dialect/VhloOps.h"  // from @stablehlo  // IWYU pragma: keep
#include "machina/compiler/mlir/machina/ir/tf_ops.h"
#include "machina/compiler/mlir/machina/transforms/passes.h"
#include "machina/compiler/mlir/machina/utils/stablehlo_custom_call.h"
#include "machina/compiler/mlir/machina/utils/xla_call_module_attrs.h"
#include "machina/compiler/tf2xla/kernels/xla_call_module_loader.h"
#include "machina/xla/mlir_hlo/mhlo/IR/hlo_ops.h"  // IWYU pragma: keep
#include "machina/xla/tsl/platform/statusor.h"

namespace mlir {
namespace TF {
namespace {

#define GEN_PASS_DEF_XLACALLMODULEDESERIALIZATIONPASS
#include "machina/compiler/mlir/machina/transforms/tf_passes.h.inc"

// `tf.backend_config` is a DictionaryAttr, JAX2TF sets the value of its
// i64 attribute `called_index` to the TF function's name.
constexpr toolchain::StringRef kTfBackendConfigAttrName = "tf.backend_config";
constexpr toolchain::StringRef kCalledIndexAttrName = "called_index";
constexpr toolchain::StringRef kCalledFuncAttrName = "called_func";

// Deserialize the StableHLO module embedded in XlaCallModuleOp's module
// attribute.
absl::StatusOr<OwningOpRef<ModuleOp>> DeserializeStablehlo(MLIRContext *context,
                                                           XlaCallModuleOp op) {
  std::vector<std::string> disabled_checks;
  for (auto attr : op.getDisabledChecks().getAsRange<StringAttr>()) {
    disabled_checks.push_back(attr.getValue().str());
  }
  std::vector<std::string> platforms;
  for (auto attr : op.getPlatforms().getAsRange<StringAttr>()) {
    platforms.push_back(attr.getValue().str());
  }

  // Set the deserialized StableHLO version to an attribute of the XlaCallModule
  // op, this is used if when the module is re-serialized.
  auto version = stablehlo::getPortableArtifactVersion(op.getModule());
  if (failed(version)) {
    return absl::InvalidArgumentError(
        "Failed to get the deserialized StableHLO version, XlaCallModuleOp "
        "must have a valid StableHLO module serialized using "
        "stablehlo::serializePortableArtifact APIs.");
  }
  Builder builder(context);
  op->setAttr(kStablehloVersionAttrName,
              builder.getStringAttr(version.value().toString()));

  TF_ASSIGN_OR_RETURN(
      auto loader,
      machina::XlaCallModuleLoader::Create(
          context, static_cast<int>(op.getVersion()), op.getModule(),
          std::move(disabled_checks), std::move(platforms),
          /*num_invocation_args=*/op.getArgs().size(),
          op.getHasTokenInputOutput(), op.getUseShardyPartitioner()));
  return std::move(*loader).module();
}

// Renames functions in the stablehlo module to avoid naming conflicts with
// existing functions in the tf module.
// Sets _from_xla_call_module attribute for each stablehlo function.
// Returns the new stablehlo main function's name or error.
//
// If we directly insert stablehlo functions into tf module, MLIR will rename
// the stablehlo functions themselves in the tf module automatically to avoid
// naming conflicts. But we need to rename the function calls inside the
// stablehlo functions as well. So we first do this renaming in the stablehlo
// module itself without inserting into the tf module.
FailureOr<StringAttr> RenameStablehloFunctions(
    MLIRContext *context, SymbolTableCollection &symbol_tables,
    ModuleOp tf_module, ModuleOp stablehlo_module) {
  SymbolTable &tf_symbol_table = symbol_tables.getSymbolTable(tf_module);
  // `stablehlo_module` is deleted right after the deserialization, so no need
  // to store its `SymbolTable` to `SymbolTableCollection`.
  SymbolTable stablehlo_symbol_table(stablehlo_module);

  Builder builder(context);
  StringAttr main_func_name;
  for (auto func : stablehlo_module.getOps<func::FuncOp>()) {
    const bool is_main_func = func.getSymName() == kStablehloMainFunctionName;
    if (tf_symbol_table.lookup(func.getSymName())) {
      if (failed(stablehlo_symbol_table.renameToUnique(
              func, {&tf_symbol_table, &stablehlo_symbol_table}))) {
        return func.emitError()
               << "failed to rename StableHLO function " << func.getSymName();
      }
    }
    if (is_main_func) {
      main_func_name = func.getSymNameAttr();
    }
    func->setAttr(kFromXlaCallModuleAttrName, builder.getUnitAttr());
  }
  if (!main_func_name) {
    return stablehlo_module.emitError()
           << "StableHLO module does not have an entry function";
  }
  return main_func_name;
}

// Moves functions from one module to another.
// The moved functions are set to private.
void MoveFunctions(SymbolTableCollection &symbol_tables, ModuleOp from,
                   ModuleOp to) {
  SymbolTable &to_symbol_table = symbol_tables.getSymbolTable(to);
  for (auto func : toolchain::make_early_inc_range(from.getOps<func::FuncOp>())) {
    func->remove();
    func.setPrivate();
    to_symbol_table.insert(func);
  }
}

void CopyStablehloModuleAttrs(ModuleOp stablehlo_module, XlaCallModuleOp op) {
  op->setAttr(kStablehloModuleAttrsAttrName,
              stablehlo_module->getAttrDictionary());
}

// Symbolizes `called_index` attributes in custom all ops to `called_func`.
LogicalResult SymbolizeCustomCallCalledIndex(
    ModuleOp module, toolchain::ArrayRef<SymbolRefAttr> function_list) {
  WalkResult result =
      module.walk([&](stablehlo::CustomCallOp op) {
        if (!IsTfFuncCustomCall(op)) {
          return WalkResult::advance();
        }

        auto backend_config =
            op->getAttrOfType<DictionaryAttr>(kTfBackendConfigAttrName);
        if (!backend_config) {
          op->emitOpError()
              << "is missing attribute '" << kTfBackendConfigAttrName << "'";
          return WalkResult::interrupt();
        }

        auto called_index_attr = mlir::dyn_cast_or_null<IntegerAttr>(
            backend_config.get(kCalledIndexAttrName));
        if (!called_index_attr) {
          op->emitOpError()
              << "is missing attribute '" << kCalledIndexAttrName << "'";
          return WalkResult::interrupt();
        }
        int called_index = called_index_attr.getInt();
        if (called_index < 0 || called_index >= function_list.size()) {
          op->emitOpError()
              << "references function #" << called_index
              << " but enclosing XlaCallModule has a function list of size "
              << function_list.size();
          return WalkResult::interrupt();
        }

        toolchain::SmallVector<NamedAttribute> new_config;
        // Copy the attributes in the current config except `called_index`.
        for (auto attr : backend_config) {
          if (attr.getName() != kCalledIndexAttrName) {
            new_config.push_back(attr);
          }
        }

        Builder builder(op.getContext());
        // Sets the `called_index` attribute to the TF function's name.
        new_config.push_back(builder.getNamedAttr(kCalledFuncAttrName,
                                                  function_list[called_index]));

        // Sets the `tf.backend_config` attribute to the `new_config`.
        op->setAttr(kTfBackendConfigAttrName,
                    builder.getDictionaryAttr(new_config));

        return WalkResult::advance();
      });
  return result.wasInterrupted() ? failure() : success();
}

LogicalResult DeserializeXlaCallModule(MLIRContext *context,
                                       SymbolTableCollection &symbol_tables,
                                       ModuleOp module, XlaCallModuleOp op) {
  auto deserialized = DeserializeStablehlo(context, op);
  if (!deserialized.ok()) {
    return op.emitOpError()
           << "failed to deserialize StableHLO module from XlaCallModule: "
           << deserialized.status().ToString();
  }
  OwningOpRef<ModuleOp> stablehlo_module = *std::move(deserialized);

  CopyStablehloModuleAttrs(*stablehlo_module, op);

  auto main_func = RenameStablehloFunctions(context, symbol_tables, module,
                                            stablehlo_module.get());
  if (failed(main_func)) {
    return failure();
  }

  // Translate `called_index` in TF function custom calls into symbol
  // references. `function_list` attribute is needed after that.
  toolchain::SmallVector<SymbolRefAttr> function_list(
      op.getFunctionList().getAsRange<SymbolRefAttr>());
  if (failed(
          SymbolizeCustomCallCalledIndex(*stablehlo_module, function_list))) {
    return failure();
  }
  op.removeFunctionListAttr();

  MoveFunctions(symbol_tables, *stablehlo_module, module);

  // Module is deserialized, we set an empty string to it instead removing
  // it because it's a required attribute.
  op.setModule("");
  // Set the stablehlo main function as a symbol attribute. This is required
  // because we not only need this to look up the StableHLO function called by
  // XlaCallModule, but also need the symbol reference to prevent DCE from
  // removing the stablehlo functions from the top-level module.
  op->setAttr(kStablehloEntryFunctionAttrName, SymbolRefAttr::get(*main_func));

  return success();
}

class XlaCallModuleDeserializationPass
    : public impl::XlaCallModuleDeserializationPassBase<
          XlaCallModuleDeserializationPass> {
 public:
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    XlaCallModuleDeserializationPassBase::getDependentDialects(registry);
    mlir::func::registerAllExtensions(registry);
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SymbolTableCollection symbol_tables;
    WalkResult result = module.walk([&](XlaCallModuleOp op) {
      if (failed(DeserializeXlaCallModule(&getContext(), symbol_tables, module,
                                          op))) {
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (result.wasInterrupted()) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>>
CreateXlaCallModuleDeserializationPass() {
  return std::make_unique<XlaCallModuleDeserializationPass>();
}

}  // namespace TF
}  // namespace mlir
