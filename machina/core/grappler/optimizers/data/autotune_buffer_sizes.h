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

#ifndef MACHINA_CORE_GRAPPLER_OPTIMIZERS_DATA_AUTOTUNE_BUFFER_SIZES_H_
#define MACHINA_CORE_GRAPPLER_OPTIMIZERS_DATA_AUTOTUNE_BUFFER_SIZES_H_

#include "absl/status/status.h"
#include "machina/core/framework/attr_value.pb.h"
#include "machina/core/grappler/optimizers/data/optimizer_base.h"

namespace machina {
namespace grappler {

constexpr char kAutotune[] = "autotune";

// This optimization does the following:
//
// 1. Adds `prefetch(AUTOTUNE)` after all asynchronous tf.data transformations
// (e.g. parallel batch, parallel map, parallel interleave, and map + batch) if
// they are not followed by a `prefetch` yet.
//
// 2. If there exists any `prefetch(buffer_size=N)` for `N>=0`,  it will replace
// the transformation with autotunable version of `prefetch` which uses N as
// the minimum size of the buffer.
class AutotuneBufferSizes : public TFDataOptimizerBase {
 public:
  AutotuneBufferSizes() = default;
  ~AutotuneBufferSizes() override = default;

  string name() const override { return "autotune_buffer_sizes"; };

  bool UsesFunctionLibrary() const override { return false; }

  absl::Status Init(
      const machina::RewriterConfig_CustomGraphOptimizer* config) override {
    if (!config) return absl::OkStatus();

    const string& autotune = config->parameter_map().at(kAutotune).s();
    if (autotune == "true") {
      autotune_ = true;
    } else if (autotune == "false") {
      autotune_ = false;
    } else {
      return absl::InvalidArgumentError(
          absl::StrCat("Received an invalid value for parameter ", kAutotune,
                       ": ", autotune));
    }
    return absl::OkStatus();
  }

  absl::Status OptimizeAndCollectStats(Cluster* cluster,
                                       const GrapplerItem& item,
                                       GraphDef* output,
                                       OptimizationStats* stats) override;

 private:
  bool autotune_ = true;
};

}  // namespace grappler
}  // namespace machina

#endif  // MACHINA_CORE_GRAPPLER_OPTIMIZERS_DATA_AUTOTUNE_BUFFER_SIZES_H_
