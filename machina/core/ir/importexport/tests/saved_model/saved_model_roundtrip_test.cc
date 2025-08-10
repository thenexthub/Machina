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

#include <string>
#include <utility>

#include "machina/core/framework/graph_debug_info.pb.h"
#include "machina/core/ir/importexport/savedmodel_export.h"
#include "machina/core/ir/importexport/savedmodel_import.h"
#include "machina/core/ir/importexport/tests/roundtrip/roundtrip.h"
#include "machina/core/platform/env.h"
#include "machina/core/platform/path.h"
#include "machina/core/platform/test.h"
#include "machina/core/protobuf/meta_graph.pb.h"
#include "machina/core/protobuf/saved_model.pb.h"

namespace {

absl::Status ReadModelProto(const std::string& input_file,
                            machina::SavedModel* out) {
  return machina::ReadBinaryProto(machina::Env::Default(), input_file,
                                     out);
}

void RunRoundTrip(const std::string& input_file) {
  mlir::DialectRegistry registry;
  mlir::MLIRContext context(registry);

  machina::SavedModel original_model;
  auto read_result = ReadModelProto(input_file, &original_model);
  ASSERT_TRUE(read_result.ok());

  machina::GraphDebugInfo debug_info;
  absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> module_ref_status =
      mlir::tfg::ImportSavedModelToMlir(&context, debug_info, original_model);

  mlir::OwningOpRef<mlir::ModuleOp> module_ref =
      std::move(module_ref_status.value());

  machina::SavedModel final_model;
  auto status = mlir::tfg::ExportMlirToSavedModel(*module_ref, original_model,
                                                  &final_model);
  if (!status.ok()) {
    LOG(ERROR) << "Export failed: " << status;
  }
  ASSERT_TRUE(status.ok()) << status;

  machina::MetaGraphDef* original_metagraph =
      original_model.mutable_meta_graphs(0);
  machina::MetaGraphDef* final_metagraph =
      final_model.mutable_meta_graphs(0);

  // In order to compare graph defs, make sure that both original and
  // final graph defs are normalized, e.g, control input are alphabetically
  // sorted.
  machina::NormalizeTensorData(*original_metagraph->mutable_graph_def(),
                                  /*add_fulltype=*/true);
  machina::NormalizeTensorData(*final_metagraph->mutable_graph_def(),
                                  /*add_fulltype=*/false);

  if (!machina::protobuf::util::MessageDifferencer::Equivalent(
          original_model, final_model)) {
#if defined(PLATFORM_GOOGLE)
    // Some of the protobuf comparisons are not available in OSS.
    // This will show the diff inline.
    EXPECT_THAT(original_model, ::testing::EquivToProto(final_model));
#else

    // That's the best we could do given there is no good diff functionality.
    LOG(WARNING) << "Saved model has changed after TFG roundtrip";
#endif
  }
}

constexpr char kTestData[] = "core/ir/importexport/tests/saved_model";

TEST(SavedModelRoundTripTest, V1ModelIsIdentity) {
  const std::string input_file =
      machina::io::JoinPath(machina::testing::TensorFlowSrcRoot(),
                               kTestData, "savedmodel_v1/saved_model.pb");

  ASSERT_NO_FATAL_FAILURE(RunRoundTrip(input_file));
}

TEST(SavedModelRoundTripTest, V2ModelIsIdentity) {
  const std::string input_file =
      machina::io::JoinPath(machina::testing::TensorFlowSrcRoot(),
                               kTestData, "savedmodel_v2/saved_model.pb");

  ASSERT_NO_FATAL_FAILURE(RunRoundTrip(input_file));
}

}  // namespace
