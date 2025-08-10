/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, March 6, 2025.
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
#include "pybind11/pytypes.h"  // from @pybind11
#include "machina/xla/tsl/util/stat_summarizer_options.h"
#include "machina/core/framework/graph.pb.h"
#include "machina/core/framework/step_stats.pb.h"
#include "machina/core/util/stat_summarizer.h"

namespace py = pybind11;

PYBIND11_MODULE(_pywrap_stat_summarizer, m) {
  py::class_<machina::StatSummarizer> stat_summ_class(m, "StatSummarizer",
                                                         py::dynamic_attr());
  stat_summ_class
      .def(py::init([](std::string graph_def_serialized) {
        machina::GraphDef proto;
        proto.ParseFromString(graph_def_serialized);
        return new machina::StatSummarizer(proto);
      }))
      .def(py::init([]() {
        return new machina::StatSummarizer(tsl::StatSummarizerOptions());
      }))
      .def("ProcessStepStats", &machina::StatSummarizer::ProcessStepStats)
      .def("GetOutputString", &machina::StatSummarizer::GetOutputString)
      .def("PrintStepStats", &machina::StatSummarizer::PrintStepStats)
      .def("ProcessStepStatsStr", [](machina::StatSummarizer& self,
                                     const std::string& step_stats_str) {
        machina::StepStats step_stats;
        step_stats.ParseFromString(step_stats_str);
        self.ProcessStepStats(step_stats);
      });
};
