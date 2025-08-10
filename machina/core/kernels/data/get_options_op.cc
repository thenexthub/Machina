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
#include "machina/core/kernels/data/get_options_op.h"

#include "absl/memory/memory.h"
#include "machina/core/data/name_utils.h"
#include "machina/core/framework/dataset.h"
#include "machina/core/framework/dataset_options.pb.h"
#include "machina/core/framework/partial_tensor_shape.h"
#include "machina/core/framework/tensor.h"
#include "machina/core/profiler/lib/traceme.h"

namespace machina {
namespace data {

void GetOptionsOp::Compute(OpKernelContext* ctx) {
  DatasetBase* input;
  OP_REQUIRES_OK(ctx, GetDatasetFromVariantTensor(ctx->input(0), &input));
  if (ctx->status().ok()) {
    Tensor* string_handle_t;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(0, TensorShape({}), &string_handle_t));
    string_handle_t->scalar<tstring>()() = input->options().SerializeAsString();
  }
}

string GetOptionsOp::TraceString(const OpKernelContext& ctx,
                                 bool verbose) const {
  return tsl::profiler::TraceMeOp(name_view(), type_string_view());
}

namespace {
REGISTER_KERNEL_BUILDER(Name("GetOptions").Device(DEVICE_CPU).Priority(2),
                        GetOptionsOp);
REGISTER_KERNEL_BUILDER(Name("GetOptions")
                            .Device(DEVICE_GPU)
                            .HostMemory("input_dataset")
                            .HostMemory("serialized_options")
                            .Priority(1),
                        GetOptionsOp);
}  // namespace
}  // namespace data
}  // namespace machina
