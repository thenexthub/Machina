/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, June 15, 2025.
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
#include "machina/core/kernels/data/rewrite_dataset_op.h"

// On mobile we do not provide rewrite dataset op because not all of its
// dependencies are available there. The op is replaced with a no-op.
#if !defined(IS_MOBILE_PLATFORM)
#include <map>
#include <string>

#include "machina/core/data/rewrite_utils.h"
#include "machina/core/protobuf/rewriter_config.pb.h"

namespace machina {
namespace data {

/* static */ constexpr const char* const RewriteDatasetOp::kDatasetType;
/* static */ constexpr const char* const RewriteDatasetOp::kInputDataset;
/* static */ constexpr const char* const RewriteDatasetOp::kRewriteName;
/* static */ constexpr const char* const RewriteDatasetOp::kOutputTypes;
/* static */ constexpr const char* const RewriteDatasetOp::kOutputShapes;

RewriteDatasetOp::RewriteDatasetOp(OpKernelConstruction* ctx)
    : UnaryDatasetOpKernel(ctx) {}

void RewriteDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                                   DatasetBase** output) {
  tstring rewrite_name;
  OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, kRewriteName, &rewrite_name));

  auto config_factory = [rewrite_name]() {
    RewriterConfig rewriter_config;
    rewriter_config.add_optimizers(std::string(rewrite_name));
    rewriter_config.set_meta_optimizer_iterations(RewriterConfig::ONE);
    rewriter_config.set_fail_on_optimizer_errors(true);
    return rewriter_config;
  };

  core::RefCountPtr<DatasetBase> rewritten;
  OP_REQUIRES_OK(ctx, RewriteDataset(ctx, input, std::move(config_factory),
                                     /*record_fingerprint=*/false, &rewritten));
  *output = rewritten.release();
}

namespace {
REGISTER_KERNEL_BUILDER(Name("RewriteDataset").Device(DEVICE_CPU),
                        RewriteDatasetOp);
}  // namespace
}  // namespace data
}  // namespace machina
#else   // !IS_MOBILE_PLATFORM
namespace machina {
namespace data {

// static

RewriteDatasetOp::RewriteDatasetOp(OpKernelConstruction* ctx)
    : UnaryDatasetOpKernel(ctx) {}

void RewriteDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                                   DatasetBase** output) {
  input->Ref();
  *output = input;
}

namespace {
REGISTER_KERNEL_BUILDER(Name("RewriteDataset").Device(DEVICE_CPU),
                        RewriteDatasetOp);
}  // namespace
}  // namespace data
}  // namespace machina
#endif  // !IS_MOBILE_PLATFORM
