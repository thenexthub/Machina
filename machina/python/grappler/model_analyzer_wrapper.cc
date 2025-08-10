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

#include <memory>
#include <sstream>

#include "pybind11/pybind11.h"  // from @pybind11
#include "machina/core/grappler/grappler_item_builder.h"
#include "machina/core/protobuf/meta_graph.pb.h"
#include "machina/python/grappler/model_analyzer.h"
#include "machina/python/lib/core/pybind11_status.h"

namespace py = pybind11;

PYBIND11_MODULE(_pywrap_model_analyzer, m) {
  m.def("GenerateModelReport",
        [](const py::bytes& serialized_metagraph, bool assume_valid_feeds,
           bool debug) -> py::bytes {
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

          machina::grappler::ModelAnalyzer analyzer(*item);

          std::ostringstream os;
          machina::MaybeRaiseFromStatus(
              analyzer.GenerateReport(debug, assume_valid_feeds, os));
          return py::bytes(os.str());
        });
}
