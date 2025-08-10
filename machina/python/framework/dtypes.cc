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

#include "pybind11/detail/common.h"  // from @pybind11
#include "pybind11/pybind11.h"  // from @pybind11
#include "machina/core/framework/types.h"
#include "machina/core/framework/types.pb.h"

namespace {

inline int DataTypeId(machina::DataType dt) { return static_cast<int>(dt); }

// A variant of machina::DataTypeString which uses fixed-width names
// for floating point data types. This behavior is compatible with that of
// existing pure Python DType.
const std::string DataTypeStringCompat(machina::DataType dt) {
  switch (dt) {
    case machina::DataType::DT_HALF:
      return "float16";
    case machina::DataType::DT_HALF_REF:
      return "float16_ref";
    case machina::DataType::DT_FLOAT:
      return "float32";
    case machina::DataType::DT_FLOAT_REF:
      return "float32_ref";
    case machina::DataType::DT_DOUBLE:
      return "float64";
    case machina::DataType::DT_DOUBLE_REF:
      return "float64_ref";
    default:
      return machina::DataTypeString(dt);
  }
}

}  // namespace

namespace machina {

constexpr DataTypeSet kNumPyIncompatibleTypes =
    ToSet(DataType::DT_RESOURCE) | ToSet(DataType::DT_VARIANT);

inline bool DataTypeIsNumPyCompatible(DataType dt) {
  return !kNumPyIncompatibleTypes.Contains(dt);
}

}  // namespace machina

namespace py = pybind11;

PYBIND11_MODULE(_dtypes, m) {
  py::class_<machina::DataType>(m, "DType")
      .def(py::init([](py::object obj) {
        auto id = static_cast<int>(py::int_(obj));
        if (machina::DataType_IsValid(id) &&
            id != static_cast<int>(machina::DT_INVALID)) {
          return static_cast<machina::DataType>(id);
        }
        throw py::type_error(
            py::str("{} does not correspond to a valid machina::DataType")
                .format(id));
      }))
      .def("__int__",
           [](machina::DataType self) { return DataTypeId(self); })
      // For compatibility with pure-Python DType.
      .def_property_readonly("_type_enum", &DataTypeId)
      .def_property_readonly(
          "as_datatype_enum", &DataTypeId,
          "Returns a `types_pb2.DataType` enum value based on this data type.")

      .def_property_readonly("name",
                             [](machina::DataType self) {
#if PY_MAJOR_VERSION < 3
                               return py::bytes(DataTypeStringCompat(self));
#else
                               return DataTypeStringCompat(self);
#endif
                             })
      .def_property_readonly(
          "size",
          [](machina::DataType self) {
            return machina::DataTypeSize(machina::BaseType(self));
          })

      .def("__repr__",
           [](machina::DataType self) {
             return py::str("tf.{}").format(DataTypeStringCompat(self));
           })
      .def("__str__",
           [](machina::DataType self) {
             return py::str("<dtype: {!r}>")
#if PY_MAJOR_VERSION < 3
                 .format(py::bytes(DataTypeStringCompat(self)));
#else
                 .format(DataTypeStringCompat(self));
#endif
           })
      .def("__hash__", &DataTypeId)

      .def_property_readonly(
          "is_numpy_compatible",
          [](machina::DataType self) {
            return machina::DataTypeIsNumPyCompatible(
                machina::BaseType(self));
          },
          "Returns whether this data type has a compatible NumPy data type.")

      .def_property_readonly(
          "is_bool",
          [](machina::DataType self) {
            return machina::BaseType(self) == machina::DT_BOOL;
          },
          "Returns whether this is a boolean data type.")
      .def_property_readonly(
          "is_numeric",
          [](machina::DataType self) {
            return machina::DataTypeIsNumeric(machina::BaseType(self));
          },
          "Returns whether this is a numeric data type.")
      .def_property_readonly(
          "is_complex",
          [](machina::DataType self) {
            return machina::DataTypeIsComplex(machina::BaseType(self));
          },
          "Returns whether this is a complex floating point type.")
      .def_property_readonly(
          "is_floating",
          [](machina::DataType self) {
            return machina::DataTypeIsFloating(machina::BaseType(self));
          },
          "Returns whether this is a (non-quantized, real) floating point "
          "type.")
      .def_property_readonly(
          "is_integer",
          [](machina::DataType self) {
            return machina::DataTypeIsInteger(machina::BaseType(self));
          },
          "Returns whether this is a (non-quantized) integer type.")
      .def_property_readonly(
          "is_quantized",
          [](machina::DataType self) {
            return machina::DataTypeIsQuantized(machina::BaseType(self));
          },
          "Returns whether this is a quantized data type.")
      .def_property_readonly(
          "is_unsigned",
          [](machina::DataType self) {
            return machina::DataTypeIsUnsigned(machina::BaseType(self));
          },
          R"doc(Returns whether this type is unsigned.

Non-numeric, unordered, and quantized types are not considered unsigned, and
this function returns `False`.)doc");

  py::implicitly_convertible<py::int_, machina::DataType>();
}
