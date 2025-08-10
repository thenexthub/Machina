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
#ifndef MACHINA_CORE_TFRT_SAVED_MODEL_SAVED_MODEL_IMPORT_INPUT_H_
#define MACHINA_CORE_TFRT_SAVED_MODEL_SAVED_MODEL_IMPORT_INPUT_H_

#include <memory>
#include <string>

#include "machina/compiler/mlir/machina/translate/import_model.h"
#include "machina/core/tfrt/fallback/fallback_state.h"
#include "machina/core/tfrt/graph_executor/config.h"
#include "machina/core/tfrt/utils/tfrt_graph_execution_state.h"

namespace machina {
namespace tfrt_stub {

// TfrtSavedModelMLIRImportInput implements SavedModelMLIRImportInput, so that
// it can perform customization (eg. Placer and Grappler) on the input graph to
// the MLIR importer.
class TfrtSavedModelMLIRImportInput : public SavedModelMLIRImportInput {
 public:
  static absl::StatusOr<TfrtSavedModelMLIRImportInput> Create(
      const FallbackState& fallback_state, const MetaGraphDef* meta_graph_def,
      const GraphDebugInfo& debug_info,
      bool run_placer_grappler_on_nested_functions = false,
      machina::tfrt_stub::RuntimeConfig* runtime_config = nullptr);

  TfrtSavedModelMLIRImportInput(
      const MetaGraphDef* meta_graph_def, const GraphDebugInfo& debug_info,
      std::unique_ptr<TfrtGraphExecutionState> graph_execution_state);

  absl::StatusOr<const machina::Graph*> GetSubGraph(
      absl::string_view name, GraphImportConfig& graph_import_config) override;

  // Return the time used by grappler.
  absl::Duration GetGrapplerDuration() const { return grappler_duration_; }

  // Return the time used by functionalization.
  absl::Duration GetFunctionalizationDuration() const {
    return functionalization_duration_;
  }

 private:
  std::unique_ptr<TfrtGraphExecutionState> graph_execution_state_;
  absl::flat_hash_map<std::string, std::unique_ptr<machina::Graph>>
      optimized_graphs_;

  absl::Duration functionalization_duration_;
  absl::Duration grappler_duration_;
};

}  // namespace tfrt_stub
}  // namespace machina

#endif  // MACHINA_CORE_TFRT_SAVED_MODEL_SAVED_MODEL_IMPORT_INPUT_H_
