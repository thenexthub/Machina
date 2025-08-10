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

#include "machina/core/ir/importexport/savedmodel_export.h"

#include <utility>

#include "machina/core/ir/importexport/graphdef_export.h"
#include "machina/core/platform/errors.h"
#include "machina/core/platform/status.h"
#include "machina/core/protobuf/meta_graph.pb.h"

namespace mlir {
namespace tfg {

absl::Status ExportMlirToSavedModel(
    mlir::ModuleOp module, const machina::SavedModel &original_saved_model,
    machina::SavedModel *output_saved_model) {
  if (original_saved_model.meta_graphs_size() == 0) {
    return machina::errors::InvalidArgument(
        "Original saved model has no meta graphs");
  }

  machina::GraphDef new_graphdef;
  TF_RETURN_WITH_CONTEXT_IF_ERROR(ConvertToGraphDef(module, &new_graphdef),
                                  "while converting TFG to GraphDef");

  // Overwrite the graph def portion of the saved model with the new one.
  machina::MetaGraphDef meta_graph_def = original_saved_model.meta_graphs(0);
  *(meta_graph_def.mutable_graph_def()) = std::move(new_graphdef);
  *output_saved_model = original_saved_model;
  *(output_saved_model->mutable_meta_graphs(0)) = std::move(meta_graph_def);

  return absl::OkStatus();
}

}  // namespace tfg
}  // namespace mlir
