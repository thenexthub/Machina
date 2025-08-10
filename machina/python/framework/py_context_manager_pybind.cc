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
#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/stl.h"  // from @pybind11
#include "machina/python/framework/py_context_manager.h"

namespace py = pybind11;

namespace {

// Test harness for PyContextManager.  Creates a PyContextManager `cm` that
// wraps `context_manager`, calls `cm.Enter()`, and then calls `body_func`
// with `cm.var()`.  Returns the result of the function.
py::handle TestPyContextManager(py::handle context_manager,
                                py::handle body_func) {
  machina::Safe_PyObjectPtr result;
  {
    machina::PyContextManager cm;
    Py_INCREF(context_manager.ptr());  // cm.Enter steals a reference.
    if (!cm.Enter(context_manager.ptr())) {
      throw py::error_already_set();
    }
    result.reset(
        PyObject_CallFunctionObjArgs(body_func.ptr(), cm.var(), nullptr));
  }
  // cm gets destroyed here.

  if (result) {
    return result.release();
  } else {
    throw py::error_already_set();
  }
}

}  // namespace

PYBIND11_MODULE(_py_context_manager, m) {
  m.def("test_py_context_manager", TestPyContextManager);
}
