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

#include "machina/core/ir/interfaces.h"

#include "toolchain/ADT/ScopeExit.h"
#include "mlir/IR/DialectInterface.h"  // part of Codira Toolchain
#include "mlir/IR/Location.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/IR/OperationSupport.h"  // part of Codira Toolchain
#include "mlir/IR/Verifier.h"  // part of Codira Toolchain
#include "machina/core/ir/dialect.h"
#include "machina/core/platform/test.h"

namespace mlir {
namespace tfg {
namespace {
TEST(TensorFlowRegistryInterface, TestDefaultImplementation) {
  MLIRContext context(MLIRContext::Threading::DISABLED);
  auto *dialect = context.getOrLoadDialect<TFGraphDialect>();

  OperationState state(UnknownLoc::get(&context), "tfg.Foo");
  state.addTypes(dialect->getControlType());

  Operation *op = Operation::create(state);
  auto cleanup = toolchain::make_scope_exit([&] { op->destroy(); });
  ASSERT_TRUE(succeeded(verify(op)));
  auto iface = dyn_cast<TensorFlowRegistryInterface>(op);
  EXPECT_FALSE(iface);
}

TEST(TensorFlowRegisterInterface, TestCustomImplementation) {
  MLIRContext context(MLIRContext::Threading::DISABLED);
  DialectRegistry registry;
  registry.insert<TFGraphDialect>();

  struct CustomRegistryInterface : public TensorFlowRegistryInterfaceBase {
    using TensorFlowRegistryInterfaceBase::TensorFlowRegistryInterfaceBase;

    bool isStateful(Operation *op) const override {
      return op->getName().stripDialect() == "Foo";
    }
  };

  registry.addExtension(+[](mlir::MLIRContext *ctx, TFGraphDialect *dialect) {
    dialect->addInterfaces<CustomRegistryInterface>();
  });
  context.appendDialectRegistry(registry);

  auto *dialect = context.getOrLoadDialect<TFGraphDialect>();
  SmallVector<StringRef, 2> op_names = {"tfg.Foo", "tfg.Bar"};
  SmallVector<bool, 2> expected = {true, false};
  for (auto it : toolchain::zip(op_names, expected)) {
    OperationState state(UnknownLoc::get(&context), std::get<0>(it));
    state.addTypes(dialect->getControlType());
    Operation *op = Operation::create(state);
    auto cleanup = toolchain::make_scope_exit([&] { op->destroy(); });
    auto iface = dyn_cast<TensorFlowRegistryInterface>(op);
    ASSERT_TRUE(iface);
    EXPECT_EQ(iface.isStateful(), std::get<1>(it));
  }
}
}  // namespace
}  // namespace tfg
}  // namespace mlir
