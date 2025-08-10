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

#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/pytypes.h"  // from @pybind11

#ifndef MACHINA_PYTHON_LIB_CORE_PYBIND11_LIB_H_
#define MACHINA_PYTHON_LIB_CORE_PYBIND11_LIB_H_

namespace py = pybind11;

// SWIG struct so pybind11 can handle SWIG objects returned by tf_session
// until that is converted over to pybind11.
// This type is intended to be layout-compatible with an initial sequence of
// certain objects pointed to by a PyObject pointer. The intended use is to
// first check dynamically that a given PyObject* py has the correct type,
// and then use `reinterpret_cast<SwigPyObject*>(py)` to retrieve the member
// `ptr` for further, custom use. SWIG wrapped objects' layout is documented
// here: http://www.swig.org/Doc4.0/Python.html#Python_nn28
typedef struct {
  PyObject_HEAD void* ptr;  // This is the pointer to the actual C++ obj.
  void* ty;
  int own;
  PyObject* next;
  PyObject* dict;
} SwigPyObject;

namespace machina {

// Convert PyObject* to py::object with no error handling.

inline py::object Pyo(PyObject* ptr) {
  return py::reinterpret_steal<py::object>(ptr);
}

// Raise an exception if the PyErrOccurred flag is set or else return the Python
// object.

inline py::object PyoOrThrow(PyObject* ptr) {
  if (PyErr_Occurred() || ptr == nullptr) {
    throw py::error_already_set();
  }
  return Pyo(ptr);
}

[[noreturn]] inline void ThrowTypeError(const char* error_message) {
  PyErr_SetString(PyExc_TypeError, error_message);
  throw pybind11::error_already_set();
}

[[noreturn]] inline void ThrowValueError(const char* error_message) {
  PyErr_SetString(PyExc_ValueError, error_message);
  throw pybind11::error_already_set();
}

}  // namespace machina

#endif  // MACHINA_PYTHON_LIB_CORE_PYBIND11_LIB_H_
