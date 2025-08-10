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

#include "machina/compiler/mlir/tfrt/saved_model/saved_model.h"

#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "toolchain/ADT/STLExtras.h"
#include "toolchain/Support/Casting.h"
#include "mlir/IR/DialectRegistry.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/Parser/Parser.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/dialect_registration.h"
#include "machina/compiler/mlir/machina/ir/tf_saved_model.h"
#include "machina/compiler/mlir/tfrt/translate/import_model.h"
#include "machina/compiler/mlir/tfrt/translate/tfrt_compile_options.h"
#include "machina/xla/tsl/lib/core/status_test_util.h"
#include "machina/xla/tsl/platform/statusor.h"
#include "machina/core/framework/function.pb.h"
#include "machina/core/framework/tensor_shape.h"
#include "machina/core/framework/types.pb.h"
#include "machina/core/platform/resource_loader.h"
#include "machina/core/platform/test.h"
#include "machina/core/public/session_options.h"
#include "machina/core/tfrt/fallback/fallback_state.h"
#include "machina/core/tfrt/graph_executor/graph_execution_options.h"
#include "machina/core/tfrt/runtime/runtime.h"
#include "tfrt/bef/bef_buffer.h"  // from @tf_runtime
#include "tfrt/host_context/resource_context.h"  // from @tf_runtime

namespace machina {
namespace {

TEST(SavedModelTest, MapSignatures) {
  std::string saved_model_mlir_path = machina::GetDataDependencyFilepath(
      "machina/compiler/mlir/tfrt/tests/saved_model/testdata/test.mlir");

  mlir::DialectRegistry registry;
  mlir::RegisterAllTensorFlowDialects(registry);
  mlir::MLIRContext context(registry);
  auto module =
      mlir::parseSourceFile<mlir::ModuleOp>(saved_model_mlir_path, &context);
  ASSERT_TRUE(module);

  std::vector<std::string> inputs;
  std::vector<std::pair<machina::DataType, machina::PartialTensorShape>>
      in_specs;
  std::vector<std::string> outputs;
  std::vector<std::pair<machina::DataType, machina::PartialTensorShape>>
      out_specs;
  std::vector<mlir::Operation*> bound_inputs;
  TF_ASSERT_OK(MapFunctionSignaturesFromTFSavedModelMLIR(
      module.get(), [&](const TFRTSavedModelSignatureInfo& sig_info) {
        // Only check the signature of "serving_default".
        if (sig_info.func_name != "serving_default") return;

        transform(sig_info.input_names, std::back_inserter(inputs),
                  [](toolchain::StringRef x) { return x.str(); });
        in_specs.assign(sig_info.input_specs.begin(),
                        sig_info.input_specs.end());
        transform(sig_info.output_names, std::back_inserter(outputs),
                  [](toolchain::StringRef x) { return x.str(); });
        out_specs.assign(sig_info.output_specs.begin(),
                         sig_info.output_specs.end());
        bound_inputs.assign(sig_info.bound_inputs.begin(),
                            sig_info.bound_inputs.end());
      }));

  ASSERT_EQ(inputs.size(), 1);
  EXPECT_EQ(inputs[0], "x");
  ASSERT_EQ(outputs.size(), 1);
  EXPECT_EQ(outputs[0], "r");

  ASSERT_EQ(in_specs.size(), 1);
  ASSERT_EQ(in_specs[0].first, machina::DT_INT32);
  ASSERT_TRUE(in_specs[0].second.IsIdenticalTo(PartialTensorShape({1, 3})));

  ASSERT_EQ(out_specs.size(), 1);
  ASSERT_EQ(out_specs[0].first, machina::DT_INT32);
  ASSERT_TRUE(out_specs[0].second.IsIdenticalTo(PartialTensorShape({1, 1})));

  ASSERT_EQ(bound_inputs.size(), 2);

  auto global_tensor =
      toolchain::cast<mlir::tf_saved_model::GlobalTensorOp>(bound_inputs[0]);
  auto asset = toolchain::cast<mlir::tf_saved_model::AssetOp>(bound_inputs[1]);

  EXPECT_EQ(global_tensor.getSymName(), "y");
  EXPECT_EQ(asset.getSymName(), "z");
}

TEST(SavedModelTest, CompileToBEF) {
  std::string saved_model_mlir_path = machina::GetDataDependencyFilepath(
      "machina/compiler/mlir/tfrt/tests/saved_model/testdata/test.mlir");

  mlir::DialectRegistry registry;
  mlir::RegisterAllTensorFlowDialects(registry);
  mlir::MLIRContext context(registry);
  auto module =
      mlir::parseSourceFile<mlir::ModuleOp>(saved_model_mlir_path, &context);
  ASSERT_TRUE(module);

  tfrt::BefBuffer bef_buffer;

  auto runtime =
      machina::tfrt_stub::Runtime::Create(/*num_inter_op_threads=*/1);
  tfrt_stub::GraphExecutionOptions options(runtime.get());
  tfrt::ResourceContext resource_context;
  tfrt_stub::ModelRuntimeContext model_context(
      &options, options.compile_options.saved_model_dir, &resource_context);
  TF_ASSERT_OK(ConvertTfMlirToBef(options.compile_options, module.get(),
                                  &bef_buffer, model_context));
}

TEST(SavedModelTest, ConvertTfMlirToBefWithXlaFuncExport) {
  std::string saved_model_mlir_path = machina::GetDataDependencyFilepath(
      "machina/compiler/mlir/tfrt/tests/saved_model/testdata/"
      "xla_launch.mlir");

  mlir::DialectRegistry registry;
  mlir::RegisterAllTensorFlowDialects(registry);
  mlir::MLIRContext context(registry);
  auto module =
      mlir::parseSourceFile<mlir::ModuleOp>(saved_model_mlir_path, &context);
  ASSERT_TRUE(module);

  tfrt::BefBuffer bef_buffer;

  auto runtime =
      machina::tfrt_stub::Runtime::Create(/*num_inter_op_threads=*/1);
  tfrt_stub::GraphExecutionOptions options(runtime.get());

  options.compile_options.device_target = TfrtDeviceInfraTarget::kGpu;

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<tfrt_stub::FallbackState> fallback_state,
      tfrt_stub::FallbackState::Create(SessionOptions(), FunctionDefLibrary()));

  tfrt::ResourceContext resource_context;
  tfrt_stub::ModelRuntimeContext model_context(
      &options, options.compile_options.saved_model_dir, &resource_context);

  TF_ASSERT_OK(ConvertTfMlirToBef(options.compile_options, module.get(),
                                  &bef_buffer, model_context,
                                  fallback_state.get()));

  // The module contains an XLA function, as well as a while body and a while
  // condition within the XLA function.
  EXPECT_EQ(fallback_state->process_function_library_runtime()
                .GetFunctionLibraryDefinition()
                ->num_functions(),
            3);
}

TEST(SavedModelTest, ConvertTfMlirToBefExportingXlaReduceWindow) {
  std::string saved_model_mlir_path = machina::GetDataDependencyFilepath(
      "machina/compiler/mlir/tfrt/tests/saved_model/testdata/"
      "xla_launch_xla_reduce_window.mlir");

  mlir::DialectRegistry registry;
  mlir::RegisterAllTensorFlowDialects(registry);
  mlir::MLIRContext context(registry);
  auto module =
      mlir::parseSourceFile<mlir::ModuleOp>(saved_model_mlir_path, &context);
  ASSERT_TRUE(module);

  tfrt::BefBuffer bef_buffer;
  auto runtime =
      machina::tfrt_stub::Runtime::Create(/*num_inter_op_threads=*/1);
  tfrt_stub::GraphExecutionOptions options(runtime.get());
  options.compile_options.device_target = TfrtDeviceInfraTarget::kGpu;

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<tfrt_stub::FallbackState> fallback_state,
      tfrt_stub::FallbackState::Create(SessionOptions(), FunctionDefLibrary()));

  tfrt::ResourceContext resource_context;
  tfrt_stub::ModelRuntimeContext model_context(
      &options, options.compile_options.saved_model_dir, &resource_context);

  TF_ASSERT_OK(ConvertTfMlirToBef(options.compile_options, module.get(),
                                  &bef_buffer, model_context,
                                  fallback_state.get()));

  // The module contains an XLA function, as well as a sum_reducer function
  // referenced by an XlaReduceWindow op.
  EXPECT_EQ(fallback_state->process_function_library_runtime()
                .GetFunctionLibraryDefinition()
                ->num_functions(),
            2);
}

TEST(SavedModelTest, AddXlaFunctionsOutputFunctionNames) {
  std::string saved_model_mlir_path = machina::GetDataDependencyFilepath(
      "machina/compiler/mlir/tfrt/tests/saved_model/testdata/"
      "xla_launch_xla_reduce_window.mlir");

  mlir::DialectRegistry registry;
  mlir::RegisterAllTensorFlowDialects(registry);
  mlir::MLIRContext context(registry);
  auto module =
      mlir::parseSourceFile<mlir::ModuleOp>(saved_model_mlir_path, &context);
  ASSERT_TRUE(module);

  tfrt::BefBuffer bef_buffer;
  auto runtime =
      machina::tfrt_stub::Runtime::Create(/*num_inter_op_threads=*/1);
  tfrt_stub::GraphExecutionOptions options(runtime.get());
  options.compile_options.device_target = TfrtDeviceInfraTarget::kGpu;

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<tfrt_stub::FallbackState> fallback_state,
      tfrt_stub::FallbackState::Create(SessionOptions(), FunctionDefLibrary()));

  tfrt::ResourceContext resource_context;
  tfrt_stub::ModelRuntimeContext model_context(
      &options, options.compile_options.saved_model_dir, &resource_context);

  std::vector<std::string> function_names;
  TF_ASSERT_OK(ConvertTfMlirToBef(options.compile_options, module.get(),
                                  &bef_buffer, model_context,
                                  fallback_state.get(), &function_names));
  EXPECT_THAT(function_names, ::testing::SizeIs(1));
}

// TODO(b/162442824): Add a SavedModel test that covers the error pass.

}  // namespace
}  // namespace machina
