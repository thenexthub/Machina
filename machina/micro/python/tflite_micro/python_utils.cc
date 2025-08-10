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

#include "python/tflite_micro/python_utils.h"

#include <Python.h>

namespace tflite {

int ConvertFromPyString(PyObject* obj, char** data, Py_ssize_t* length) {
#if PY_MAJOR_VERSION >= 3
  if (PyUnicode_Check(obj)) {
    // const_cast<> is for CPython 3.7 finally adding const to the API.
    *data = const_cast<char*>(PyUnicode_AsUTF8AndSize(obj, length));
    return *data == nullptr ? -1 : 0;
  }
  return PyBytes_AsStringAndSize(obj, data, length);
#else
  return PyString_AsStringAndSize(obj, data, length);
#endif
}

PyObject* ConvertToPyString(const char* data, size_t length) {
#if PY_MAJOR_VERSION >= 3
  return PyBytes_FromStringAndSize(data, length);
#else
  return PyString_FromStringAndSize(data, length);
#endif
}

}  // namespace tflite
