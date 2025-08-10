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
#ifndef MACHINA_CORE_TFRT_SAVED_MODEL_PYTHON_SAVED_MODEL_LOAD_AND_RUN_H_
#define MACHINA_CORE_TFRT_SAVED_MODEL_PYTHON_SAVED_MODEL_LOAD_AND_RUN_H_

#include <Python.h>

#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "machina/core/framework/tensor.h"
#include "machina/core/lib/core/status.h"
#include "machina/core/platform/statusor.h"
#include "machina/core/tfrt/graph_executor/graph_execution_options.h"
#include "machina/core/tfrt/saved_model/saved_model.h"

namespace machina::tfrt_stub {

absl::StatusOr<std::unique_ptr<SavedModel>> LoadSavedModel(
    absl::string_view saved_model_dir,
    const std::unordered_set<std::string>& tags);

std::vector<machina::Tensor> RunConvertor(PyObject* args);

absl::Status Run(
    SavedModel* saved_model,
    const machina::tfrt_stub::GraphExecutionRunOptions& run_options,
    absl::string_view name, const std::vector<machina::Tensor>& inputs,
    std::vector<machina::Tensor>* outputs);
}  // namespace machina::tfrt_stub

#endif  // MACHINA_CORE_TFRT_SAVED_MODEL_PYTHON_SAVED_MODEL_LOAD_AND_RUN_H_
