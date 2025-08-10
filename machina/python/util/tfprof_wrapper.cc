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

#include "pybind11/pybind11.h"  // from @pybind11
#include "machina/core/profiler/internal/print_model_analysis.h"

namespace py = pybind11;

PYBIND11_MODULE(_pywrap_tfprof, m) {
  m.def("PrintModelAnalysis",
        [](const std::string* graph, const std::string* run_meta,
           const std::string* op_log, const std::string* command,
           const std::string* options) {
          std::string temp = machina::tfprof::PrintModelAnalysis(
              graph, run_meta, op_log, command, options);
          return py::bytes(temp);
        });
  m.def("NewProfiler", &machina::tfprof::NewProfiler);
  m.def("ProfilerFromFile", &machina::tfprof::ProfilerFromFile);
  m.def("DeleteProfiler", &machina::tfprof::DeleteProfiler);
  m.def("AddStep", &machina::tfprof::AddStep);
  m.def("SerializeToString", []() {
    std::string temp = machina::tfprof::SerializeToString();
    return py::bytes(temp);
  });
  m.def("WriteProfile", &machina::tfprof::WriteProfile);
  m.def("Profile", [](const std::string* command, const std::string* options) {
    std::string temp = machina::tfprof::Profile(command, options);
    return py::bytes(temp);
  });
}
