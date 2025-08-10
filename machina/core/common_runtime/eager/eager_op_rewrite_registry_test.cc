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

#include "machina/core/common_runtime/eager/eager_op_rewrite_registry.h"

#include <memory>

#include "machina/core/platform/test.h"

namespace machina {

class TestEagerOpRewrite : public EagerOpRewrite {
 public:
  TestEagerOpRewrite(string name, string file, string line)
      : EagerOpRewrite(name, file, line),
        executor_(/*async=*/false, /*enable_streaming_enqueue=*/true) {}
  static int count_;
  EagerExecutor executor_;
  absl::Status Run(
      EagerOperation* orig_op,
      std::unique_ptr<machina::EagerOperation>* out_op) override {
    ++count_;
    // Create a new NoOp Eager operation.
    machina::EagerOperation* op =
        new machina::EagerOperation(&orig_op->EagerContext());
    TF_RETURN_IF_ERROR(op->Reset("NoOp", nullptr, false, &executor_));
    out_op->reset(op);
    return absl::OkStatus();
  }
};

int TestEagerOpRewrite::count_ = 0;

// Register two rewriter passes during the PRE_EXECUTION phase
REGISTER_REWRITE(EagerOpRewriteRegistry::PRE_EXECUTION, 10000,
                 TestEagerOpRewrite);
REGISTER_REWRITE(EagerOpRewriteRegistry::PRE_EXECUTION, 10001,
                 TestEagerOpRewrite);

TEST(EagerOpRewriteRegistryTest, RegisterRewritePass) {
  EXPECT_EQ(0, TestEagerOpRewrite::count_);
  StaticDeviceMgr device_mgr(DeviceFactory::NewDevice(
      "CPU", {}, "/job:localhost/replica:0/task:0/device:CPU:0"));
  machina::EagerContext* ctx = new machina::EagerContext(
      SessionOptions(),
      machina::ContextDevicePlacementPolicy::DEVICE_PLACEMENT_SILENT, false,
      &device_mgr, false, nullptr, nullptr, nullptr,
      /*run_eager_op_as_function=*/true);
  EagerOperation orig_op(ctx);
  std::unique_ptr<machina::EagerOperation> out_op;
  EXPECT_EQ(absl::OkStatus(),
            EagerOpRewriteRegistry::Global()->RunRewrite(
                EagerOpRewriteRegistry::PRE_EXECUTION, &orig_op, &out_op));
  EXPECT_EQ(2, TestEagerOpRewrite::count_);
  EXPECT_EQ("NoOp", out_op->Name());
  ctx->Unref();
}

}  // namespace machina
