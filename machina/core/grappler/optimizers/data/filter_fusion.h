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

#ifndef MACHINA_CORE_GRAPPLER_OPTIMIZERS_DATA_FILTER_FUSION_H_
#define MACHINA_CORE_GRAPPLER_OPTIMIZERS_DATA_FILTER_FUSION_H_

#include "machina/core/grappler/optimizers/data/optimizer_base.h"

namespace machina {
namespace grappler {

// This optimization fuses filter transformations.
class FilterFusion : public TFDataOptimizerBase {
 public:
  FilterFusion() = default;
  ~FilterFusion() override = default;

  string name() const override { return "filter_fusion"; };

  bool UsesFunctionLibrary() const override { return false; }

  absl::Status Init(
      const machina::RewriterConfig_CustomGraphOptimizer* config) override {
    return absl::OkStatus();
  }

  absl::Status OptimizeAndCollectStats(Cluster* cluster,
                                       const GrapplerItem& item,
                                       GraphDef* output,
                                       OptimizationStats* stats) override;
};

}  // namespace grappler
}  // namespace machina

#endif  // MACHINA_CORE_GRAPPLER_OPTIMIZERS_DATA_FILTER_FUSION_H_
