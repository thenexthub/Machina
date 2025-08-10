
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

#include "machina/dtensor/mlir/spmd_expander.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "toolchain/ADT/DenseMap.h"
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_ops.h"
#include "machina/core/platform/errors.h"
#include "machina/dtensor/cc/dstatus.h"
#include "machina/dtensor/cc/tensor_layout.h"

namespace machina {
namespace dtensor {
namespace {

using ::testing::IsNull;
using ::testing::NotNull;

class DummyExpander : public SPMDExpanderBase {
  StatusOr<mlir::Operation*> ExpandOp(mlir::Operation* op) override {
    return errors::Unimplemented("");
  }

  StatusOr<toolchain::DenseMap<int, Layout>> ComputeLayoutForward(
      mlir::Operation* op,
      const toolchain::DenseMap<int, Layout>& input_layouts) override {
    return errors::Unimplemented("");
  }
  StatusOr<toolchain::DenseMap<int, Layout>> ComputeLayoutBackward(
      mlir::Operation* op,
      const toolchain::DenseMap<int, Layout>& output_layouts) override {
    return errors::Unimplemented("");
  }
};

class SPMDExpanderRegistryTest : public ::testing::Test {
 public:
  SPMDExpanderRegistryTest() {
    registry_.RegisterPropagateFn(mlir::TF::AddOp::getOperationName().str(),
                                  std::make_unique<DummyExpander>());
  }

 protected:
  SPMDExpanderRegistry registry_;
};

TEST_F(SPMDExpanderRegistryTest, LookupFromOpName) {
  EXPECT_THAT(registry_.GetPropagateFnForFullOpName("tf.Add"), NotNull());
  EXPECT_THAT(registry_.GetPropagateFnForFullOpName("Unknown"), IsNull());
}

}  // namespace
}  // namespace dtensor
}  // namespace machina
