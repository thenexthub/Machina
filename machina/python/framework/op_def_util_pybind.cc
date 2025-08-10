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
#include "machina/python/framework/op_def_util.h"

namespace py = pybind11;

namespace {

py::handle ConvertAttr(py::handle value, std::string attr_type) {
  machina::Safe_PyObjectPtr result =
      ::machina::ConvertPyObjectToAttributeType(
          value.ptr(), ::machina::AttributeTypeFromName(attr_type));
  if (!result) {
    throw py::error_already_set();
  }
  Py_INCREF(result.get());
  return result.release();
}

py::handle SerializedAttrValueToPyObject(std::string attr_value_string) {
  machina::AttrValue attr_value;
  attr_value.ParseFromString(attr_value_string);
  machina::Safe_PyObjectPtr result =
      ::machina::AttrValueToPyObject(attr_value);
  if (!result) {
    throw py::error_already_set();
  }
  Py_INCREF(result.get());
  return result.release();
}

}  // namespace

// Expose op_def_util.h functions via Python.
PYBIND11_MODULE(_op_def_util, m) {
  // Note: the bindings below are added for testing purposes; but the functions
  // are expected to be called from c++, not Python.
  m.def("ConvertPyObjectToAttributeType", ConvertAttr, py::arg("value"),
        py::arg("attr_type_enum"));
  m.def("SerializedAttrValueToPyObject", SerializedAttrValueToPyObject,
        py::arg("attr_value_string"));
}
