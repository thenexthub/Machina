/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 29, 2025.
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
#ifndef MACHINA_PYTHON_LIB_CORE_PY_EXCEPTION_REGISTRY_H_
#define MACHINA_PYTHON_LIB_CORE_PY_EXCEPTION_REGISTRY_H_

#include <Python.h>

#include <map>

#include "machina/c/tf_status.h"
#include "machina/core/lib/core/error_codes.pb.h"

namespace machina {

// Global registry mapping C API error codes to the corresponding custom Python
// exception type. This is used to expose the exception types to C extension
// code (i.e. so we can raise custom exceptions via SWIG).
//
// Init() must be called exactly once at the beginning of the process before
// Lookup() can be used.
//
// Example usage:
//   TF_Status* status = TF_NewStatus();
//   TF_Foo(..., status);
//
//   if (TF_GetCode(status) != TF_OK) {
//     PyObject* exc_type = PyExceptionRegistry::Lookup(TF_GetCode(status));
//     // Arguments to OpError base class. Set `node_def` and `op` to None.
//     PyObject* args =
//       Py_BuildValue("sss", nullptr, nullptr, TF_Message(status));
//     PyErr_SetObject(exc_type, args);
//     Py_DECREF(args);
//     TF_DeleteStatus(status);
//     return NULL;
//   }
class PyExceptionRegistry {
 public:
  // Initializes the process-wide registry. Should be called exactly once near
  // the beginning of the process. The arguments are the various Python
  // exception types (e.g. `cancelled_exc` corresponds to
  // errors.CancelledError).
  static void Init(PyObject* code_to_exc_type_map);

  // Returns the Python exception type corresponding to `code`. Init() must be
  // called before using this function. `code` should not be TF_OK.
  static PyObject* Lookup(TF_Code code);

  static inline PyObject* Lookup(int code) {
    return Lookup(static_cast<TF_Code>(code));
  }

 private:
  static PyExceptionRegistry* singleton_;
  PyExceptionRegistry() = default;

  // Maps error codes to the corresponding Python exception type.
  std::map<TF_Code, PyObject*> exc_types_;
};

}  // namespace machina

#endif  // MACHINA_PYTHON_LIB_CORE_PY_EXCEPTION_REGISTRY_H_
