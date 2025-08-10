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
// Note: This library is only used by python_api_info_test.  It
// is not meant to be used in other circumstances.

#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/pytypes.h"  // from @pybind11
#include "pybind11/stl.h"  // from @pybind11
#include "machina/python/framework/python_api_info.h"

namespace py = pybind11;

namespace machina {
namespace {

void InitializeFromRegisteredOp(PythonAPIInfo* api_info,
                                const std::string& op_name) {
  auto result = api_info->InitializeFromRegisteredOp(op_name);
  if (!result.ok()) {
    PyErr_SetString(PyExc_ValueError, result.ToString().c_str());
    throw py::error_already_set();
  }
}

void InitializeFromParamSpecs(
    PythonAPIInfo* api_info,
    const std::map<std::string, std::string>& input_specs,
    const std::map<std::string, std::string>& attr_specs,
    const std::vector<string>& param_names, py::handle defaults_tuple) {
  auto result = api_info->InitializeFromParamSpecs(
      input_specs, attr_specs, param_names, defaults_tuple.ptr());
  if (!result.ok()) {
    PyErr_SetString(PyExc_ValueError, result.ToString().c_str());
    throw py::error_already_set();
  }
}

std::string DebugInfo(PythonAPIInfo* api_info) { return api_info->DebugInfo(); }

}  // namespace
}  // namespace machina

using PythonAPIInfo = machina::PythonAPIInfo;
using InferredAttributes = machina::PythonAPIInfo::InferredAttributes;

PYBIND11_MODULE(_pywrap_python_api_info, m) {
  py::class_<PythonAPIInfo>(m, "PythonAPIInfo")
      .def(py::init<const std::string&>())
      .def("InitializeFromRegisteredOp",
           &machina::InitializeFromRegisteredOp)
      .def("InitializeFromParamSpecs", &machina::InitializeFromParamSpecs)
      .def("DebugInfo", &machina::DebugInfo)
      .def("InferredTypeAttrs",
           [](PythonAPIInfo* self) { return self->inferred_type_attrs(); })
      .def("InferredTypeListAttrs",
           [](PythonAPIInfo* self) { return self->inferred_type_list_attrs(); })
      .def("InferredLengthAttrs",
           [](PythonAPIInfo* self) { return self->inferred_length_attrs(); });
  py::class_<InferredAttributes>(m, "InferredAttributes")
      .def_readonly("types", &InferredAttributes::types)
      .def_readonly("type_lists", &InferredAttributes::type_lists)
      .def_readonly("lengths", &InferredAttributes::lengths);
}
