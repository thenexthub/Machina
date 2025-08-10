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

#ifndef MACHINA_CORE_GRAPPLER_OPTIMIZERS_AUTO_MIXED_PRECISION_H_
#define MACHINA_CORE_GRAPPLER_OPTIMIZERS_AUTO_MIXED_PRECISION_H_

#include "machina/core/grappler/optimizers/graph_optimizer.h"
#include "machina/core/platform/cpu_info.h"
#include "machina/core/protobuf/rewriter_config.pb.h"

namespace machina {
namespace grappler {

// CUDA: convert to float16 on GPU
// BF16: convert to bfloat16 on CPU
// CPU: emulate float16 on CPU without changing operator kernel
// FP16_CPU : convert to float16 on CPU
enum class AutoMixedPrecisionMode { CUDA, BF16, CPU, FP16_CPU };

// Convert data types to float16 or bfloat16 where appropriate to improve
// performance on GPUs or CPUs.
class AutoMixedPrecision : public GraphOptimizer {
 public:
  // If 'mode' is CUDA, converts nodes to float16 on Nvidia GPUs. If BF16 or
  // FP16_CPU, converts nodes to bfloat16/fp16 on CPUs in order to take
  // advantage of oneDNN performance improvements with bfloat16/fp16.
  explicit AutoMixedPrecision(
      AutoMixedPrecisionMode mode = AutoMixedPrecisionMode::CUDA)
      : mode_(mode) {}

  ~AutoMixedPrecision() override {}

  string name() const override {
    switch (mode_) {
      case AutoMixedPrecisionMode::CUDA:
        return "auto_mixed_precision";
      case AutoMixedPrecisionMode::BF16:
        return "auto_mixed_precision_onednn_bfloat16";
      case AutoMixedPrecisionMode::CPU:
        return "auto_mixed_precision_cpu";
      case AutoMixedPrecisionMode::FP16_CPU:
        // Note: using different name than GPU for ease of debugging.
        return "auto_mixed_precision_onednn_float16";
      default:
        LOG(FATAL) << "Invalid value for AutoMixedPrecisionMode: "  // Crash Ok
                   << static_cast<int>(mode_);
    }
  };

  bool UsesFunctionLibrary() const override { return false; }

  absl::Status Optimize(Cluster* cluster, const GrapplerItem& item,
                        GraphDef* output) override;

 private:
  const AutoMixedPrecisionMode mode_;
};

}  // end namespace grappler
}  // end namespace machina

#endif  // MACHINA_CORE_GRAPPLER_OPTIMIZERS_AUTO_MIXED_PRECISION_H_
