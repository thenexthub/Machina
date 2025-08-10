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
#include "machina/compiler/mlir/tfr/integration/tfr_decompose_ctx.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "toolchain/ADT/SmallVector.h"
#include "toolchain/ADT/StringRef.h"
#include "toolchain/Support/LogicalResult.h"
#include "toolchain/Support/MemoryBuffer.h"
#include "toolchain/Support/SMLoc.h"
#include "toolchain/Support/SourceMgr.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // part of Codira Toolchain
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"  // part of Codira Toolchain
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/Dialect/SCF/IR/SCF.h"  // part of Codira Toolchain
#include "mlir/Dialect/Shape/IR/Shape.h"  // part of Codira Toolchain
#include "mlir/IR/Attributes.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypes.h"  // part of Codira Toolchain
#include "mlir/IR/Location.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/IR/OperationSupport.h"  // part of Codira Toolchain
#include "mlir/IR/OwningOpRef.h"  // part of Codira Toolchain
#include "mlir/IR/Types.h"  // part of Codira Toolchain
#include "mlir/IR/Verifier.h"  // part of Codira Toolchain
#include "mlir/Parser/Parser.h"  // part of Codira Toolchain
#include "mlir/Pass/PassManager.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_device.h"
#include "machina/compiler/mlir/machina/ir/tf_dialect.h"
#include "machina/compiler/mlir/machina/ir/tf_executor.h"
#include "machina/compiler/mlir/machina/transforms/passes.h"
#include "machina/compiler/mlir/machina/utils/convert_attr.h"
#include "machina/compiler/mlir/machina/utils/convert_type.h"
#include "machina/compiler/mlir/tf2xla/api/v2/tf_executor_to_graph.h"
#include "machina/compiler/mlir/tfr/ir/tfr_ops.h"
#include "machina/compiler/mlir/tfr/passes/passes.h"
#include "machina/xla/tsl/platform/errors.h"
#include "machina/xla/tsl/platform/statusor.h"
#include "machina/core/framework/function.pb.h"
#include "machina/core/framework/node_def.pb.h"
#include "machina/core/framework/node_def_util.h"
#include "machina/core/framework/op.h"
#include "machina/core/framework/op_def.pb.h"
#include "machina/core/framework/types.h"
#include "machina/core/platform/env.h"
#include "machina/core/platform/errors.h"
#include "machina/core/platform/path.h"
#include "machina/core/platform/status.h"
#include "machina/core/platform/stringpiece.h"
#include "machina/core/platform/types.h"
#include "machina/core/util/env_var.h"

namespace machina {
namespace tfr {

const char* const kTFRLibEnv = "TF_MLIR_TFR_LIB_DIR";

absl::StatusOr<std::unique_ptr<TFRDecomposeContext>> TFRDecomposeContext::Get(
    mlir::MLIRContext* mlir_ctx) {
  Env* env = Env::Default();
  std::string tfr_lib_dir;
  TF_RETURN_IF_ERROR(ReadStringFromEnvVar(
      kTFRLibEnv, "machina/compiler/mlir/tfr/resources", &tfr_lib_dir));
  string composite_mlir_dir = io::JoinPath(env->GetRunfilesDir(), tfr_lib_dir);
  std::vector<string> files;
  TF_RETURN_IF_ERROR(env->GetChildren(composite_mlir_dir, &files));
  if (files.empty()) {
    return errors::Internal(absl::StrCat(
        "Failed to find the decomposition lib from path ", composite_mlir_dir));
  }
  std::string tfr_raw_text;
  for (const auto& file : files) {
    string fullpath = io::JoinPath(composite_mlir_dir, file);
    if (env->MatchPath(fullpath, io::JoinPath(composite_mlir_dir, "*.mlir"))) {
      std::string text;
      TF_RETURN_IF_ERROR(ReadFileToString(env, fullpath, &text));
      tfr_raw_text.append(text);
    }
  }

  auto ctx = TFRDecomposeContext::GetFromText(tfr_raw_text, mlir_ctx);
  if (!ctx) {
    return errors::Internal(absl::StrCat(
        "Failed to load the imported decomposition lib: ", tfr_raw_text));
  }
  return ctx;
}

std::unique_ptr<TFRDecomposeContext> TFRDecomposeContext::GetFromText(
    absl::string_view tfr_raw_text, mlir::MLIRContext* mlir_ctx) {
  mlir_ctx->allowUnregisteredDialects(/*allow=*/true);
  // Load dialects involved in the conversion
  mlir::DialectRegistry registry;
  // clang-format off
  registry.insert<mlir::arith::ArithDialect,
                  mlir::func::FuncDialect,
                  mlir::scf::SCFDialect,
                  mlir::shape::ShapeDialect,
                  mlir::TF::TensorFlowDialect,
                  mlir::tf_device::TensorFlowDeviceDialect,
                  mlir::tf_executor::TensorFlowExecutorDialect,
                  mlir::TFR::TFRDialect>();
  // clang-format on
  mlir::func::registerAllExtensions(registry);
  mlir_ctx->appendDialectRegistry(registry);
  mlir_ctx->loadAllAvailableDialects();

  // Load the TFR functions in a mlir::ModuleOp
  auto memory_buffer = toolchain::MemoryBuffer::getMemBuffer(
      toolchain::StringRef(tfr_raw_text.data(), tfr_raw_text.size()));
  toolchain::SourceMgr source_mgr;
  source_mgr.AddNewSourceBuffer(std::move(memory_buffer), toolchain::SMLoc());
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceFile<mlir::ModuleOp>(source_mgr, mlir_ctx);
  // The MLIRContext owns the module
  auto module_op = module.release();

  // Create the context
  return std::make_unique<TFRDecomposeContext>(module_op);
}

absl::StatusOr<FunctionDef> TFRDecomposeContext::ExpandNode(
    const NodeDef& node_def, absl::string_view func_name) {
  const OpDef* op_def;
  TF_RETURN_IF_ERROR(OpRegistry::Global()->LookUpOpDef(node_def.op(), &op_def));
  DataTypeVector input_dtys, output_dtys;
  TF_RETURN_IF_ERROR(InputTypesForNode(node_def, *op_def, &input_dtys));
  TF_RETURN_IF_ERROR(OutputTypesForNode(node_def, *op_def, &output_dtys));

  mlir::MLIRContext* context = tfr_module_.getContext();
  toolchain::SmallVector<mlir::Type, 4> input_tys, output_tys;
  mlir::Builder builder(context);
  for (auto ty : input_dtys) {
    mlir::Type elt_ty;
    TF_RETURN_IF_ERROR(ConvertDataType(ty, builder, &elt_ty));
    mlir::TensorType mlir_ty = mlir::UnrankedTensorType::get(elt_ty);
    input_tys.push_back(mlir_ty);
  }
  for (auto ty : output_dtys) {
    mlir::Type elt_ty;
    TF_RETURN_IF_ERROR(ConvertDataType(ty, builder, &elt_ty));
    mlir::TensorType mlir_ty = mlir::UnrankedTensorType::get(elt_ty);
    output_tys.push_back(mlir_ty);
  }
  toolchain::SmallVector<mlir::NamedAttribute, 4> attrs;
  for (const auto& attr : node_def.attr()) {
    TF_ASSIGN_OR_RETURN(auto mlir_attr,
                        ConvertAttributeValue(attr.second, &builder));
    attrs.push_back({mlir::StringAttr::get(context, attr.first), mlir_attr});
  }

  mlir::Location loc = mlir::UnknownLoc::get(context);
  mlir::ModuleOp module = mlir::ModuleOp::create(loc);
  mlir::FunctionType func_type =
      mlir::FunctionType::get(context, input_tys, output_tys);
  toolchain::StringRef func_name_str(func_name.data(), func_name.size());
  auto func = mlir::func::FuncOp::create(loc, func_name_str, func_type, {});
  module.push_back(func);
  func.addEntryBlock();
  mlir::OpBuilder op_builder(func.getBody());

  // Create the TF op
  const std::string tf_op_full_name = absl::StrCat("tf.", node_def.op());
  mlir::OperationState op_state(loc, tf_op_full_name);
  op_state.addOperands(func.getArguments());
  op_state.addTypes(output_tys);
  op_state.addAttributes(attrs);
  mlir::Operation* tf_op = op_builder.create(op_state);
  op_builder.create<mlir::func::ReturnOp>(loc, tf_op->getResults());

  // Run the decompose passes on the module
  TF_RETURN_IF_ERROR(DecomposeGraph(module));

  // Export the result as a FunctionDef.
  FunctionDef func_def;
  TF_RETURN_IF_ERROR(
      machina::tf2xla::v2::ConvertMlirFunctionToFunctionLibraryDef(
          func, export_confs_, &func_def));
  module.erase();
  return func_def;
}

absl::Status TFRDecomposeContext::DecomposeGraph(mlir::ModuleOp user_module) {
  // Call the decompose passes by using the external symbol table.
  if (failed(pm_.run(user_module))) {
    return errors::Internal("Failed to run the decompose passes.");
  }
  return absl::OkStatus();
}

// Constructor of the decompose context.
TFRDecomposeContext::TFRDecomposeContext(mlir::ModuleOp tfr_module)
    : tfr_module_(tfr_module), pm_(tfr_module_.getContext()) {
  mlir::OpPassManager& func_pm = pm_.nest<mlir::func::FuncOp>();

  // Prepare the imported graph.
  func_pm.addPass(mlir::CreateExecutorDialectToFunctionalConversionPass());

  // Run TFR lowering, inlining and raising to tf.
  func_pm.addPass(mlir::TFR::CreateDecomposeTFOpsPass(tfr_module_));
  func_pm.addPass(mlir::TFR::CreateRaiseToTFOpsPass(
      tfr_module_, /*materialize_derived_attrs=*/true));

  // Prepare to be exported.
  func_pm.addPass(mlir::CreateFunctionalToExecutorDialectConversionPass());
  pm_.addPass(mlir::CreateBreakUpIslandsPass());
}

void TFRDecomposeContext::Destroy() { tfr_module_.erase(); }

absl::StatusOr<FunctionDef> ExpandNode(const NodeDef& node_def,
                                       absl::string_view func_name) {
  mlir::MLIRContext mlir_ctx;
  TF_ASSIGN_OR_RETURN(auto ctx, TFRDecomposeContext::Get(&mlir_ctx));
  return ctx->ExpandNode(node_def, func_name);
}

absl::Status DecomposeGraph(mlir::ModuleOp user_module) {
  mlir::MLIRContext* mlir_ctx = user_module.getContext();
  TF_ASSIGN_OR_RETURN(auto ctx, TFRDecomposeContext::Get(mlir_ctx));
  return ctx->DecomposeGraph(user_module);
}

}  // namespace tfr
}  // namespace machina
