/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Saturday, June 21, 2025.
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

#include "machina/core/kernels/batch_kernel_test_util.h"

#include <vector>

#include "machina/xla/tsl/platform/status.h"
#include "machina/core/framework/node_def_builder.h"
#include "machina/core/kernels/batch_kernels.h"
#include "machina/core/platform/status.h"

namespace machina {
namespace test_util {

BatchFunctionKernelTestAccess::BatchFunctionKernelTestAccess(
    const BatchFunctionKernel* kernel)
    : kernel_(kernel) {}

bool BatchFunctionKernelTestAccess::enable_adaptive_batch_threads() const {
  return kernel_->enable_adaptive_batch_threads_;
}

absl::Status BatchFunctionKernelTestBase::Init(bool enable_adaptive_scheduler) {
  std::vector<DataType> input_dtypes({DataType::DT_INT64, DataType::DT_INT64});
  std::vector<NodeDefBuilder::NodeOut> inputs(
      {NodeDefBuilder::NodeOut({"n1", 0, DataType::DT_INT64}),
       NodeDefBuilder::NodeOut({"n2", 1, DataType::DT_INT64})});
  NameAttrList f;
  f.set_name("func_to_batch");
  TF_CHECK_OK(NodeDefBuilder("BatchTPUInput", "BatchFunction")
                  .Attr("max_batch_size", 32)
                  .Attr("num_batch_threads", enable_adaptive_scheduler ? 0 : 8)
                  .Attr("allowed_batch_sizes", {2, 4, 8})
                  .Attr("batch_timeout_micros", 1000)
                  .Attr("max_enqueued_batches", 100)
                  .Attr("enable_large_batch_splitting", true)
                  .Attr("low_priority_max_batch_size", 64)
                  .Attr("low_priority_batch_timeout_micros", 8000)
                  .Attr("low_priority_allowed_batch_sizes", {32, 64})
                  .Attr("low_priority_max_enqueued_batches", 1000)
                  .Attr("Tcaptured", std::vector<DataType>{DataType::DT_INT64})
                  .Attr("Tin", input_dtypes)
                  .Input(inputs)
                  .Attr("Tcaptured", std::vector<DataType>{DataType::DT_INT64})
                  .Input(std::vector<NodeDefBuilder::NodeOut>{
                      NodeDefBuilder::NodeOut({"n3", 1, DataType::DT_INT64})})
                  .Attr("Tout", std::vector<DataType>(4, DataType::DT_INT64))
                  .Attr("f", f)
                  .Finalize(node_def()));
  return InitOp();
}

}  // namespace test_util
}  // namespace machina
