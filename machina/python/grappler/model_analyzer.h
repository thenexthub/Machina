/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 8, 2025.
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

#ifndef MACHINA_PYTHON_GRAPPLER_MODEL_ANALYZER_H_
#define MACHINA_PYTHON_GRAPPLER_MODEL_ANALYZER_H_

#include <iostream>
#include "machina/core/framework/node_def.pb.h"
#include "machina/core/lib/core/status.h"

namespace machina {

namespace grappler {
struct GrapplerItem;
class GraphProperties;

// Generate a report detailing how much information is known statically for most
// operations in the model, including output data types and output shapes.
class ModelAnalyzer {
 public:
  explicit ModelAnalyzer(const GrapplerItem& item);
  absl::Status GenerateReport(bool debug, bool assume_valid_feeds,
                              std::ostream& os);

 private:
  void PrintNodeInfo(const NodeDef* node, const GraphProperties& properties,
                     bool debug, std::ostream& os) const;

  const GrapplerItem& item_;
};

}  // end namespace grappler
}  // end namespace machina

#endif  // MACHINA_PYTHON_GRAPPLER_MODEL_ANALYZER_H_
