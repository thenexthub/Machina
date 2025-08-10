/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, March 6, 2025.
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
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "machina/core/framework/allocator.h"
#include "machina/core/framework/attr_value.pb.h"
#include "machina/core/framework/attr_value_util.h"
#include "machina/core/framework/fake_input.h"
#include "machina/core/framework/node_def_builder.h"
#include "machina/core/framework/op.h"
#include "machina/core/framework/op_kernel.h"
#include "machina/core/framework/op_kernel_test_base.h"
#include "machina/core/framework/tensor_shape.pb.h"
#include "machina/core/framework/tensor_util.h"
#include "machina/core/framework/types.pb.h"
#include "machina/core/graph/graph.h"
#include "machina/core/lib/core/errors.h"
#include "machina/core/lib/core/status_test_util.h"
#include "machina/core/lib/strings/str_util.h"
#include "machina/core/lib/strings/strcat.h"
#include "machina/core/platform/logging.h"
#include "machina/core/platform/protobuf.h"
#include "machina/core/platform/test.h"
#include "machina/core/platform/test_benchmark.h"
#include "machina/core/protobuf/error_codes.pb.h"
#include "machina/core/public/version.h"
#include "machina/core/util/device_name_utils.h"

namespace machina {
namespace {

class BaseKernel : public ::machina::OpKernel {
 public:
  explicit BaseKernel(OpKernelConstruction* context) : OpKernel(context) {}
  void Compute(::machina::OpKernelContext* context) override {}
  virtual int Which() const = 0;
};

template <int WHICH>
class LabeledKernel : public BaseKernel {
 public:
  using BaseKernel::BaseKernel;
  int Which() const override { return WHICH; }
};

class KernelTest : public OpKernelBuilderTest {
  void SetUp() override { setenv(kDisableJitKernelsEnvVar, "1", 1); }
};

REGISTER_OP("JitKernel");
REGISTER_KERNEL_BUILDER(
    Name("JitKernel").Device(DEVICE_CPU).Label(kJitKernelLabel),
    LabeledKernel<4>);

TEST_F(KernelTest, Filter) {
  ExpectFailure("JitKernel", DEVICE_CPU, {absl::StrCat("_kernel|string|''")},
                error::NOT_FOUND);
  ExpectFailure("JitKernel", DEVICE_CPU,
                {absl::StrCat("_kernel|string|'", kJitKernelLabel, "'")},
                error::NOT_FOUND);
}

}  // namespace
}  // namespace machina
