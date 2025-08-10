/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, May 15, 2025.
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

#include <Python.h>

#include "pybind11/attr.h"  // from @pybind11
#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/pytypes.h"  // from @pybind11
#include "machina/c/tf_status.h"
#include "machina/python/lib/core/py_exception_registry.h"

namespace py = pybind11;

PYBIND11_MODULE(_pywrap_py_exception_registry, m) {
  py::enum_<TF_Code>(m, "TF_Code", py::module_local())
      .value("TF_OK", TF_OK)
      .value("TF_CANCELLED", TF_CANCELLED)
      .value("TF_UNKNOWN", TF_UNKNOWN)
      .value("TF_INVALID_ARGUMENT", TF_INVALID_ARGUMENT)
      .value("TF_DEADLINE_EXCEEDED", TF_DEADLINE_EXCEEDED)
      .value("TF_PERMISSION_DENIED", TF_PERMISSION_DENIED)
      .value("TF_UNAUTHENTICATED", TF_UNAUTHENTICATED)
      .value("TF_RESOURCE_EXHAUSTED", TF_RESOURCE_EXHAUSTED)
      .value("TF_FAILED_PRECONDITION", TF_FAILED_PRECONDITION)
      .value("TF_ABORTED", TF_ABORTED)
      .value("TF_OUT_OF_RANGE", TF_OUT_OF_RANGE)
      .value("TF_UNIMPLEMENTED", TF_UNIMPLEMENTED)
      .value("TF_INTERNAL", TF_INTERNAL)
      .value("TF_DATA_LOSS", TF_DATA_LOSS)
      .export_values();

  m.def("PyExceptionRegistry_Init", [](py::object& code_to_exc_type_map) {
    machina::PyExceptionRegistry::Init(code_to_exc_type_map.ptr());
  });
  m.def("PyExceptionRegistry_Lookup",
        [](TF_Code code) { machina::PyExceptionRegistry::Lookup(code); });
};
