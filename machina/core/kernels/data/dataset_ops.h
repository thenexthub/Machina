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
#ifndef MACHINA_CORE_KERNELS_DATA_DATASET_OPS_H_
#define MACHINA_CORE_KERNELS_DATA_DATASET_OPS_H_

#include <memory>

#include "machina/core/platform/platform.h"

// On mobile we do not provide this functionality because not all of its
// dependencies are available there.
#if !defined(IS_MOBILE_PLATFORM)
#include "machina/core/framework/dataset.h"
#include "machina/core/framework/op_kernel.h"

namespace machina {
namespace data {

class DatasetToGraphOp : public OpKernel {
 public:
  static constexpr const char* const kAllowStateful = "allow_stateful";
  static constexpr const char* const kStripDeviceAssignment =
      "strip_device_assignment";
  static constexpr const char* const kExternalStatePolicy =
      "external_state_policy";
  static constexpr const char* const kDatasetToGraph = "DatasetToGraph";

  explicit DatasetToGraphOp(OpKernelConstruction* ctx);

  void Compute(OpKernelContext* ctx) override;

 private:
  const int op_version_;
  ExternalStatePolicy external_state_policy_ = ExternalStatePolicy::POLICY_WARN;
  bool strip_device_assignment_ = false;
};

class DatasetCardinalityOp : public OpKernel {
 public:
  explicit DatasetCardinalityOp(OpKernelConstruction* ctx);

  void Compute(OpKernelContext* ctx) override;

 private:
  std::unique_ptr<CardinalityOptions> cardinality_options_;
};

// An OpKernel that computes the fingerprint of a dataset.
class DatasetFingerprintOp : public OpKernel {
 public:
  explicit DatasetFingerprintOp(OpKernelConstruction* ctx);

  void Compute(OpKernelContext* ctx) override;
};

class DatasetFromGraphOp : public OpKernel {
 public:
  static constexpr const char* const kGraphDef = "graph_def";
  static constexpr const char* const kHandle = "handle";

  explicit DatasetFromGraphOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override;
};

}  // namespace data
}  // namespace machina
#endif  // !IS_MOBILE_PLATFORM

#endif  // MACHINA_CORE_KERNELS_DATA_DATASET_OPS_H_
