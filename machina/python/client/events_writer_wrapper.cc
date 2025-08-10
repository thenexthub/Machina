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

#include <string>

#include "absl/status/status.h"
#include "pybind11/attr.h"  // from @pybind11
#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/pytypes.h"  // from @pybind11
#include "machina/core/lib/core/status.h"
#include "machina/core/util/events_writer.h"
#include "machina/python/lib/core/pybind11_absl.h"
#include "machina/python/lib/core/pybind11_proto.h"
#include "machina/python/lib/core/pybind11_status.h"

namespace py = pybind11;

PYBIND11_MODULE(_pywrap_events_writer, m) {
  py::class_<absl::Status> Status(m, "Status", py::module_local());
  py::class_<machina::EventsWriter> events_writer_class(m, "EventsWriter");
  events_writer_class.def(py::init<const std::string&>())
      .def("InitWithSuffix",
           [](machina::EventsWriter& self, const std::string& suffix) {
             return self.InitWithSuffix(suffix);
           })
      .def("FileName",
           [](machina::EventsWriter& self) { return self.FileName(); })
      .def("_WriteSerializedEvent",
           [](machina::EventsWriter& self, const std::string& event_str) {
             self.WriteSerializedEvent(event_str);
           })
      .def("Flush", [](machina::EventsWriter& self) { return self.Flush(); })
      .def("Close", [](machina::EventsWriter& self) { return self.Close(); })
      .def("WriteEvent",
           [](machina::EventsWriter& self, const py::object obj) {
             // Verify the proto type is an event prior to writing.
             machina::CheckProtoType(obj, "machina.Event");
             self.WriteSerializedEvent(
                 obj.attr("SerializeToString")().cast<std::string>());
           });
};
