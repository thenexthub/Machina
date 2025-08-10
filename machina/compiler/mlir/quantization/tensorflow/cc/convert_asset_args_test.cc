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
#include "machina/compiler/mlir/quantization/machina/cc/convert_asset_args.h"

#include <gmock/gmock.h>
#include "absl/strings/string_view.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/Attributes.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinAttributes.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/IR/OwningOpRef.h"  // part of Codira Toolchain
#include "mlir/Parser/Parser.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_dialect.h"
#include "machina/compiler/mlir/machina/ir/tf_saved_model.h"
#include "machina/core/platform/test.h"
#include "machina/core/protobuf/meta_graph.pb.h"

namespace mlir::quant {
namespace {

using ::machina::AssetFileDef;
using ::testing::Eq;
using ::testing::IsEmpty;
using ::testing::IsNull;
using ::testing::NotNull;
using ::testing::SizeIs;

class ConvertAssetArgsTest : public ::testing::Test {
 protected:
  ConvertAssetArgsTest() {
    ctx_.loadDialect<func::FuncDialect, TF::TensorFlowDialect,
                     tf_saved_model::TensorFlowSavedModelDialect>();
  }

  // Parses `module_op_str` to create a `ModuleOp`. Checks whether the created
  // module op is valid.
  OwningOpRef<ModuleOp> ParseModuleOpString(
      const absl::string_view module_op_str) {
    auto module_op_ref = parseSourceString<ModuleOp>(module_op_str, &ctx_);
    EXPECT_TRUE(module_op_ref);
    return module_op_ref;
  }

  mlir::MLIRContext ctx_{};
};

func::FuncOp GetMainFuncOp(ModuleOp module_op) {
  for (auto func_op : module_op.getOps<func::FuncOp>()) {
    if (func_op.getSymName() == "main") {
      return func_op;
    }
  }
  return {};
}

TEST_F(ConvertAssetArgsTest, ConvertsSingleAssetArg) {
  OwningOpRef<ModuleOp> module_op = ParseModuleOpString(R"mlir(
    module {
      "tf_saved_model.asset"() {filename = "assets/file_0.txt", sym_name = "__tf_saved_model_asset0"} : () -> ()
      func.func @main(%arg_0: tensor<!tf_type.string> {tf_saved_model.bound_input = @__tf_saved_model_asset0}) -> () attributes {tf.entry_function = {inputs = "arg_0:0", outputs = ""}} {
        return
      }
    }
  )mlir");

  FailureOr<SmallVector<AssetFileDef>> asset_file_defs =
      ConvertAssetArgs(*module_op);

  EXPECT_TRUE(succeeded(asset_file_defs));
  EXPECT_THAT(*asset_file_defs, SizeIs(1));

  const AssetFileDef& asset_file_def = *asset_file_defs->begin();
  EXPECT_THAT(asset_file_def.filename(), Eq("file_0.txt"));
  EXPECT_THAT(asset_file_def.tensor_info().name(), Eq("arg_0:0"));

  func::FuncOp main_func_op = GetMainFuncOp(*module_op);
  DictionaryAttr arg_attrs = main_func_op.getArgAttrDict(/*index=*/0);

  EXPECT_THAT(arg_attrs.get("tf_saved_model.bound_input"), IsNull());

  const ArrayRef<Attribute> index_path_attrs =
      mlir::cast<ArrayAttr>(arg_attrs.get("tf_saved_model.index_path"))
          .getValue();
  EXPECT_THAT(index_path_attrs, SizeIs(1));
  StringAttr index_path =
      mlir::dyn_cast_or_null<StringAttr>(index_path_attrs[0]);
  EXPECT_THAT(index_path, NotNull());
  EXPECT_THAT(index_path, Eq("arg_0:0"));
}

TEST_F(ConvertAssetArgsTest, NonBoundedArgsNotModified) {
  OwningOpRef<ModuleOp> module_op = ParseModuleOpString(R"mlir(
    module {
      func.func @main(%arg_0: tensor<!tf_type.string> {tf_saved_model.index_path = ["arg_0:0"]}) -> () attributes {tf.entry_function = {inputs = "arg_0:0", outputs = ""}} {
        return
      }
    }
  )mlir");

  FailureOr<SmallVector<AssetFileDef>> asset_file_defs =
      ConvertAssetArgs(*module_op);

  EXPECT_TRUE(succeeded(asset_file_defs));
  EXPECT_THAT(*asset_file_defs, IsEmpty());

  func::FuncOp main_func_op = GetMainFuncOp(*module_op);
  DictionaryAttr arg_attrs = main_func_op.getArgAttrDict(/*index=*/0);

  EXPECT_THAT(arg_attrs.get("tf_saved_model.bound_input"), IsNull());

  const ArrayRef<Attribute> index_path_attrs =
      mlir::cast<ArrayAttr>(arg_attrs.get("tf_saved_model.index_path"))
          .getValue();
  EXPECT_THAT(index_path_attrs, SizeIs(1));
  StringAttr index_path =
      mlir::dyn_cast_or_null<StringAttr>(index_path_attrs[0]);
  EXPECT_THAT(index_path, NotNull());
  EXPECT_THAT(index_path, Eq("arg_0:0"));
}

TEST_F(ConvertAssetArgsTest, ArgsBoundedToGlobalTensorNotModified) {
  // If the argument is not bound to AssetOp, it is not modified.
  OwningOpRef<ModuleOp> module_op = ParseModuleOpString(R"mlir(
    module {
      "tf_saved_model.global_tensor"() {type = tensor<2xi32>, value = dense<2> : tensor<2xi32>, sym_name = "__tf_saved_model_x"} : () -> ()
      func.func @main(%arg_0: tensor<!tf_type.resource<tensor<2xi32>>> {tf_saved_model.bound_input = @__tf_saved_model_x}) -> () attributes {tf.entry_function = {inputs = "arg_0:0", outputs = ""}} {
        return
      }
    }
  )mlir");

  FailureOr<SmallVector<AssetFileDef>> asset_file_defs =
      ConvertAssetArgs(*module_op);

  EXPECT_TRUE(succeeded(asset_file_defs));
  EXPECT_THAT(*asset_file_defs, IsEmpty());

  func::FuncOp main_func_op = GetMainFuncOp(*module_op);
  DictionaryAttr arg_attrs = main_func_op.getArgAttrDict(/*index=*/0);

  EXPECT_THAT(arg_attrs.get("tf_saved_model.bound_input"), NotNull());
}

TEST_F(ConvertAssetArgsTest, FailsWhenNoMain) {
  OwningOpRef<ModuleOp> module_op = ParseModuleOpString(R"mlir(module {})mlir");

  FailureOr<SmallVector<AssetFileDef>> asset_file_defs =
      ConvertAssetArgs(*module_op);

  EXPECT_TRUE(failed(asset_file_defs));
}

}  // namespace
}  // namespace mlir::quant
