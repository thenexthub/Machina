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
#include <string>
#include <unordered_set>
#include <vector>

#include "absl/strings/string_view.h"
#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/pytypes.h"  // from @pybind11
#include "pybind11/stl.h"  // from @pybind11
#include "pybind11_abseil/status_casters.h"  // from @pybind11_abseil
#include "machina/core/framework/tensor.h"
#include "machina/core/tfrt/graph_executor/graph_execution_options.h"
#include "machina/core/tfrt/saved_model/python/saved_model_load_and_run.h"
#include "machina/core/tfrt/saved_model/saved_model.h"
#include "machina/python/lib/core/pybind11_lib.h"

namespace py = pybind11;
namespace machina::tfrt_stub {

PYBIND11_MODULE(_pywrap_saved_model, m) {
  py::google::ImportStatusModule();

  py::class_<machina::tfrt_stub::SavedModel> give_me_a_name(m, "SavedModel");

  m.def("LoadSavedModel", &machina::tfrt_stub::LoadSavedModel,
        py::arg("saved_model_dir") = absl::string_view(),
        py::arg("tags") = std::unordered_set<std::string>());

  m.def("RunConvertor", [](const py::args args) {
    return machina::tfrt_stub::RunConvertor(args.ptr());
  });

  py::class_<machina::tfrt_stub::GraphExecutionRunOptions>(
      m, "GraphExecutionRunOptions")
      .def(py::init<>());
  m.doc() =
      "pybind11 GraphExecutionRunOptions wrapper";  // optional module docstring

  py::class_<machina::Tensor>(m, "Tensor").def(py::init<>());

  m.def("Run", &machina::tfrt_stub::Run, py::arg("saved_model") = nullptr,
        py::arg("run_options") =
            machina::tfrt_stub::GraphExecutionRunOptions(),
        py::arg("name") = absl::string_view(),
        py::arg("inputs") = std::vector<machina::Tensor>(),
        py::arg("outputs") = std::vector<machina::Tensor>());
}
}  // namespace machina::tfrt_stub
