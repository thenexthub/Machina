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

#include <memory>
#include <sstream>
#include <string>

#include "pybind11/pybind11.h"  // from @pybind11
#include "machina/core/grappler/clusters/single_machine.h"
#include "machina/core/grappler/grappler_item.h"
#include "machina/core/grappler/grappler_item_builder.h"
#include "machina/core/protobuf/meta_graph.pb.h"
#include "machina/python/grappler/cost_analyzer.h"
#include "machina/python/lib/core/pybind11_status.h"

namespace py = pybind11;

PYBIND11_MODULE(_pywrap_cost_analyzer, m) {
  m.def("GenerateCostReport",
        [](const py::bytes& serialized_metagraph, bool per_node_report,
           bool verbose, machina::grappler::Cluster* cluster) -> py::bytes {
          machina::MetaGraphDef metagraph;
          if (!metagraph.ParseFromString(std::string(serialized_metagraph))) {
            return "The MetaGraphDef could not be parsed as a valid protocol "
                   "buffer";
          }

          machina::grappler::ItemConfig cfg;
          cfg.apply_optimizations = false;
          std::unique_ptr<machina::grappler::GrapplerItem> item =
              machina::grappler::GrapplerItemFromMetaGraphDef(
                  "metagraph", metagraph, cfg);
          if (item == nullptr) {
            return "Error: failed to preprocess metagraph: check your log file "
                   "for errors";
          }

          std::string suffix;
          machina::grappler::CostAnalyzer analyzer(*item, cluster, suffix);

          std::stringstream os;
          machina::MaybeRaiseFromStatus(
              analyzer.GenerateReport(os, per_node_report, verbose));
          return py::bytes(os.str());
        });
}
