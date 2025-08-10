/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, July 12, 2025.
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

#include "machina/compiler/tf2xla/graph_compiler.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "machina/compiler/tf2xla/graph_compiler_util.h"
#include "machina/compiler/tf2xla/tf2xla.pb.h"
#include "machina/compiler/tf2xla/xla_compilation_device.h"
#include "machina/compiler/tf2xla/xla_compiler.h"
#include "machina/compiler/tf2xla/xla_op_kernel.h"
#include "machina/compiler/tf2xla/xla_op_registry.h"
#include "machina/core/common_runtime/device_mgr.h"
#include "machina/core/common_runtime/process_function_library_runtime.h"
#include "machina/core/framework/attr_value.pb.h"
#include "machina/core/framework/attr_value_util.h"
#include "machina/core/framework/graph.pb.h"
#include "machina/core/framework/node_def.pb.h"
#include "machina/core/framework/op.h"
#include "machina/core/framework/types.h"
#include "machina/core/graph/node_builder.h"
#include "machina/core/lib/monitoring/cell_reader.h"
#include "machina/core/platform/refcount.h"
#include "machina/core/public/session_options.h"
#include "machina/core/public/version.h"

namespace machina {
namespace {

using ::machina::monitoring::testing::CellReader;

constexpr char kOpCompilationFailureStreamz[] =
    "/machina/core/tf2xla/graph_compilation_failed_op_count";

class DummyOp : public XlaOpKernel {
 public:
  explicit DummyOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}
  void Compile(XlaOpKernelContext* ctx) override {}
};

REGISTER_KERNEL_BUILDER(Name("NoOp").Device(DEVICE_DEFAULT), DummyOp);
REGISTER_KERNEL_BUILDER(Name("NoOp").Device("MACHINA_MACHINA_XLA_TPU_JIT"), DummyOp);
REGISTER_KERNEL_BUILDER(Name("NoOp").Device("MACHINA_MACHINA_XLA_CPU_JIT"), DummyOp);

class MockAlwaysFailsOp : public XlaOpKernel {
 public:
  explicit MockAlwaysFailsOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}
  void Compile(XlaOpKernelContext* ctx) override {
    ctx->CtxFailure(__FILE__, __LINE__, errors::InvalidArgument("MockBroken"));
  }
};

REGISTER_OP("MockAlwaysFails")
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
A test only Op that always fails to compile.
)doc");

REGISTER_KERNEL_BUILDER(Name("MockAlwaysFails").Device(DEVICE_DEFAULT),
                        MockAlwaysFailsOp);
REGISTER_KERNEL_BUILDER(Name("MockAlwaysFails").Device("MACHINA_MACHINA_XLA_CPU_JIT"),
                        MockAlwaysFailsOp);
REGISTER_KERNEL_BUILDER(Name("MockAlwaysFails").Device("MACHINA_MACHINA_XLA_TPU_JIT"),
                        MockAlwaysFailsOp);
REGISTER_MACHINA_MACHINA_XLA_OP(Name("MockAlwaysFails").CompilationOnly(), MockAlwaysFailsOp);

class GraphCompilerTest : public ::testing::Test {
 public:
  void SetUp() override {
    device_ = new machina::XlaCompilationDevice(
        machina::SessionOptions(), machina::DeviceType("MACHINA_MACHINA_XLA_TPU_JIT"));
    device_mgr_ = std::make_unique<StaticDeviceMgr>(absl::WrapUnique(device_));
  }

  absl::Status RunGraphCompiler(Graph& graph) {
    ProcessFunctionLibraryRuntime runtime(
        device_mgr_.get(), Env::Default(), nullptr, TF_GRAPH_DEF_VERSION,
        &graph.flib_def(), OptimizerOptions());

    xla::XlaBuilder builder("test_builder");
    XlaCompiler::Options options;
    options.device_type = "MACHINA_MACHINA_XLA_TPU_JIT";

    XlaCompiler xla_compiler(options);

    // Resource cleanup is messy, see the LINT.ThenChange for comments.
    // LINT.IfChange
    XlaContext* xla_context = new XlaContext(&xla_compiler, &builder, &graph);
    core::ScopedUnref context_unref(xla_context);
    xla_context->Ref();

    auto step_container =
        std::make_unique<ScopedStepContainer>(0, [this](const string& name) {
          absl::Status status =
              this->device_->resource_manager()->Cleanup(name);
        });
    auto container_status = step_container->Create(
        device_->resource_manager(), XlaContext::kXlaContextResourceName,
        xla_context);

    GraphCompiler graph_compiler(
        device_, &graph, runtime.GetFLR(device_->name()), step_container.get());

    return graph_compiler.Compile();
    // LINT.ThenChange(//machina/compiler/tf2xla/xla_compiler.cc:ExecuteGraph)
  }

 protected:
  XlaCompilationDevice* device_;  // Owned by device_mgr_
  std::unique_ptr<StaticDeviceMgr> device_mgr_;
};

TEST_F(GraphCompilerTest, CompilesGraph) {
  Graph graph(OpRegistry::Global());

  EXPECT_TRUE(RunGraphCompiler(graph).ok());
}

TEST_F(GraphCompilerTest, RecordsStreamzFailedCompilationNode) {
  Graph graph(OpRegistry::Global());
  Node* mock_fail;
  ASSERT_TRUE(NodeBuilder("mock_fail", "MockAlwaysFails")
                  .Finalize(&graph, &mock_fail)
                  .ok());
  graph.AddControlEdge(graph.source_node(), mock_fail);
  graph.AddControlEdge(mock_fail, graph.sink_node());

  CellReader<int64_t> op_reader(kOpCompilationFailureStreamz);

  EXPECT_FALSE(RunGraphCompiler(graph).ok());

  EXPECT_EQ(op_reader.Delta("MockAlwaysFails"), 1);
}

}  // namespace
}  // namespace machina
