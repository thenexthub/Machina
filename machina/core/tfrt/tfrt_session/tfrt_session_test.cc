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
#include "machina/core/tfrt/tfrt_session/tfrt_session.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/memory/memory.h"
#include "absl/time/time.h"
#include "machina/cc/framework/ops.h"
#include "machina/cc/framework/scope.h"
#include "machina/cc/ops/array_ops.h"
#include "machina/cc/ops/const_op.h"
#include "machina/cc/saved_model/reader.h"
#include "machina/xla/tsl/lib/core/status_test_util.h"
#include "machina/core/common_runtime/device_mgr.h"
#include "machina/core/framework/graph.pb.h"
#include "machina/core/framework/tensor_shape.h"
#include "machina/core/framework/tensor_testutil.h"
#include "machina/core/framework/types.pb.h"
#include "machina/core/platform/cpu_info.h"
#include "machina/core/platform/resource_loader.h"
#include "machina/core/platform/status.h"
#include "machina/core/platform/threadpool_options.h"
#include "machina/core/protobuf/config.pb.h"
#include "machina/core/protobuf/meta_graph.pb.h"
#include "machina/core/public/session.h"
#include "machina/core/public/session_options.h"
#include "machina/core/tfrt/runtime/runtime.h"
#include "machina/core/tfrt/saved_model/saved_model_testutil.h"
#include "machina/core/tfrt/utils/thread_pool.h"
#include "tsl/platform/protobuf.h"

namespace machina {
namespace {

// Global environment for initializing the TFRT session.
class TfrtSessionEnvironment : public ::testing::Environment {
 public:
  void SetUp() override {
    TfrtSessionOptions options{
        .threadpool_options = machina::TfrtThreadpoolOptions{
            .num_main_threads = machina::port::MaxParallelism(),
            .init_timeout = absl::Milliseconds(100),
            .max_concurrent_handler = 128,
            .num_sub_thread_pool = 1}};

    TF_ASSERT_OK(InitializeTfrtSession(options));
  }
};

class TfrtSessionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create a TfrtSession.
    SessionOptions options;
    options.config.mutable_experimental()->set_use_tfrt(true);
    auto* model_metadata =
        options.config.mutable_experimental()->mutable_session_metadata();
    model_metadata->set_name("toy_v1");
    model_metadata->set_version(0);
    session_.reset(NewSession(options));
    ASSERT_TRUE(session_ != nullptr);

    // Initialize the session with a GraphDef.
    std::string saved_model_dir = GetDataDependencyFilepath(
        "machina/core/tfrt/saved_model/tests/toy_v1/1");

    MetaGraphDef meta_graph_def;
    TF_ASSERT_OK(ReadMetaGraphDefFromSavedModel(saved_model_dir, {"serve"},
                                                &meta_graph_def));

    TF_ASSERT_OK(session_->Create(meta_graph_def.graph_def()));

    // Run the init op. This also tests the case of no output tensors.
    TF_ASSERT_OK(session_->Run(/*inputs=*/{}, /*output_tensor_names=*/{},
                               /*target_tensor_names=*/{"init"}, nullptr));

    // Set up the input tensors for test cases.
    inputs_.push_back(std::make_pair(
        "input1", test::AsTensor<int32_t>({1, 1, 1}, TensorShape{1, 3})));

    inputs_.push_back(std::make_pair(
        "input2", test::AsTensor<int32_t>({2, 2, 2}, TensorShape{1, 3})));

    inputs_.push_back(std::make_pair(
        "input3", test::AsTensor<int32_t>({3, 3, 3}, TensorShape{1, 3})));
  }

  std::unique_ptr<Session> session_;

  std::vector<std::pair<std::string, Tensor>> inputs_;

  std::vector<std::string> output_tensor_names_{"result1", "result21",
                                                "result31"};
  std::vector<std::string> target_node_names_{"result22", "result32"};
};

TEST_F(TfrtSessionTest, NoTargetNodes) {
  std::vector<Tensor> outputs;
  TF_ASSERT_OK(session_->Run(inputs_, output_tensor_names_,
                             /*target_tensor_names=*/{}, &outputs));

  ASSERT_EQ(outputs.size(), 3);

  // Check output "r1".
  test::ExpectEqual(outputs[0],
                    test::AsTensor<int32_t>({6}, TensorShape{1, 1}));

  // Check output "r21".
  test::ExpectEqual(outputs[1],
                    test::AsTensor<int32_t>({12}, TensorShape{1, 1}));

  // Check output "r31".
  test::ExpectEqual(outputs[2],
                    test::AsTensor<int32_t>({18}, TensorShape{1, 1}));
}

TEST_F(TfrtSessionTest, RunOptions) {
  SessionOptions options;
  options.config.mutable_experimental()->set_use_tfrt(true);
  auto* model_metadata =
      options.config.mutable_experimental()->mutable_session_metadata();
  model_metadata->set_name("toy_v1");
  model_metadata->set_version(0);

  auto session = absl::WrapUnique(NewSession(options));
  ASSERT_TRUE(session != nullptr);

  machina::GraphDef graph_def;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(
      R"pb(
        node: {
          name: "input"
          op: "Placeholder"
          attr: {
            key: "dtype"
            value: { type: DT_INT32 }
          }
        }
        node: {
          name: "sleep_seconds"
          op: "Const"
          attr: {
            key: "dtype"
            value: { type: DT_INT32 }
          }
          attr: {
            key: "value"
            value: {
              tensor: {
                tensor_shape: {}
                dtype: DT_INT32
                int_val: 2
              }
            }
          }
        }
        node: {
          name: "sleep"
          op: "SleepIdentityOp"
          input: "sleep_seconds:0"
          input: "input:0"
          attr: {
            key: "T"
            value: { type: DT_INT32 }
          }
        })pb"

      ,
      &graph_def));

  TF_ASSERT_OK(session->Create(graph_def));

  std::vector<Tensor> outputs;
  // Test the Run() overload with RunOptions and RunMetadata
  RunMetadata run_metadata;
  TF_ASSERT_OK(session->Run(
      RunOptions{},
      /*inputs=*/{{"input", test::AsTensor<int32_t>({1}, TensorShape{1})}},
      /*output_tensor_names=*/{"sleep"},
      /*target_tensor_names=*/{}, &outputs, &run_metadata));

  ASSERT_EQ(outputs.size(), 1);

  // Check output "r1".
  test::ExpectEqual(outputs[0], test::AsTensor<int32_t>({1}, TensorShape{1}));

  // Test timeout.
  RunOptions run_options;
  run_options.set_timeout_in_ms(1);
  auto status = session->Run(
      run_options,
      /*inputs=*/{{"input", test::AsTensor<int32_t>({1}, TensorShape{1})}},
      /*output_tensor_names=*/{"sleep"},
      /*target_tensor_names=*/{}, &outputs, &run_metadata);

  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(), ::testing::HasSubstr("Deadline exceeded"));
}

TEST_F(TfrtSessionTest, ThreadPoolOptions) {
  std::vector<Tensor> outputs;
  // Test the Run() overload with RunOptions, RunMetadata, and
  // ThreadPoolOptions.
  RunMetadata run_metadata;
  tfrt_stub::TfThreadPool intra_op_thread_pool(/*name=*/"tf_intra",
                                               /*num_threads=*/1);
  tfrt_stub::TfThreadPool inter_op_thread_pool(
      /*name=*/"tf_inter",
      /*num_threads=*/1);
  thread::ThreadPoolOptions thread_pool_options{
      .inter_op_threadpool = &inter_op_thread_pool,
      .intra_op_threadpool = &intra_op_thread_pool};
  TF_ASSERT_OK(session_->Run(RunOptions{}, inputs_, output_tensor_names_,
                             /*target_tensor_names=*/{}, &outputs,
                             &run_metadata, thread_pool_options));

  ASSERT_EQ(outputs.size(), 3);

  // Check output "r1".
  test::ExpectEqual(outputs[0],
                    test::AsTensor<int32_t>({6}, TensorShape{1, 1}));
}

TEST_F(TfrtSessionTest, ThreadPoolOptions_OnlyInter) {
  std::vector<Tensor> outputs;
  // Test the Run() overload with RunOptions, RunMetadata, and
  // ThreadPoolOptions.
  RunMetadata run_metadata;
  tfrt_stub::TfThreadPool inter_op_thread_pool(
      /*name=*/"tf_inter",
      /*num_threads=*/1);
  thread::ThreadPoolOptions thread_pool_options{
      .inter_op_threadpool = &inter_op_thread_pool,
      .intra_op_threadpool = nullptr};
  TF_ASSERT_OK(session_->Run(RunOptions{}, inputs_, output_tensor_names_,
                             /*target_tensor_names=*/{}, &outputs,
                             &run_metadata, thread_pool_options));

  ASSERT_EQ(outputs.size(), 3);

  // Check output "r1".
  test::ExpectEqual(outputs[0],
                    test::AsTensor<int32_t>({6}, TensorShape{1, 1}));
}

TEST_F(TfrtSessionTest, ThreadPoolOptions_OnlyIntra) {
  std::vector<Tensor> outputs;
  // Test the Run() overload with RunOptions, RunMetadata, and
  // ThreadPoolOptions.
  RunMetadata run_metadata;
  tfrt_stub::TfThreadPool intra_op_thread_pool(/*name=*/"tf_intra",
                                               /*num_threads=*/1);
  thread::ThreadPoolOptions thread_pool_options{
      .inter_op_threadpool = nullptr,
      .intra_op_threadpool = &intra_op_thread_pool};
  TF_ASSERT_OK(session_->Run(RunOptions{}, inputs_, output_tensor_names_,
                             /*target_tensor_names=*/{}, &outputs,
                             &run_metadata, thread_pool_options));

  ASSERT_EQ(outputs.size(), 3);

  // Check output "r1".
  test::ExpectEqual(outputs[0],
                    test::AsTensor<int32_t>({6}, TensorShape{1, 1}));
}

TEST_F(TfrtSessionTest, RunInCallerThreadSessionOptions) {
  // Create a TfrtSession.
  SessionOptions options;
  options.config.mutable_experimental()->set_use_tfrt(true);
  options.config.set_inter_op_parallelism_threads(-1);
  session_.reset(NewSession(options));
  ASSERT_TRUE(session_ != nullptr);

  // Initialize the session with a GraphDef.
  std::string saved_model_dir = GetDataDependencyFilepath(
      "machina/core/tfrt/saved_model/tests/toy_v1/1");

  MetaGraphDef meta_graph_def;
  TF_ASSERT_OK(ReadMetaGraphDefFromSavedModel(saved_model_dir, {"serve"},
                                              &meta_graph_def));

  TF_ASSERT_OK(session_->Create(meta_graph_def.graph_def()));

  // Run the init op. This also tests the case of no output tensors.
  RunMetadata run_metadata;
  TF_ASSERT_OK(session_->Run(
      /*run_options=*/{}, /*inputs=*/{}, /*output_tensor_names=*/{},
      /*target_tensor_names=*/{"init"}, nullptr, &run_metadata));
}

TEST_F(TfrtSessionTest, RunInCallerThreadRunOptions) {
  std::vector<Tensor> outputs;
  // Test the Run() overload with RunOptions and RunMetadata, and run in the
  // caller thread
  RunOptions run_options;
  run_options.set_inter_op_thread_pool(-1);
  RunMetadata run_metadata;
  TF_ASSERT_OK(session_->Run(run_options, inputs_, output_tensor_names_,
                             /*target_tensor_names=*/{}, &outputs,
                             &run_metadata));

  ASSERT_EQ(outputs.size(), 3);

  // Check output "r1".
  test::ExpectEqual(outputs[0],
                    test::AsTensor<int32_t>({6}, TensorShape{1, 1}));
}

TEST_F(TfrtSessionTest, DeviceManager) {
  // Create a TfrtSession.
  SessionOptions options;
  options.config.mutable_experimental()->set_use_tfrt(true);
  options.config.set_inter_op_parallelism_threads(-1);
  session_.reset(NewSession(options));
  ASSERT_TRUE(session_ != nullptr);

  const DeviceMgr* device_manager;
  TF_ASSERT_OK(session_->LocalDeviceManager(&device_manager));

  // Initialize the session with a GraphDef.
  std::string saved_model_dir = GetDataDependencyFilepath(
      "machina/core/tfrt/saved_model/tests/toy_v1/1");

  MetaGraphDef meta_graph_def;
  TF_ASSERT_OK(ReadMetaGraphDefFromSavedModel(saved_model_dir, {"serve"},
                                              &meta_graph_def));

  TF_ASSERT_OK(session_->Create(meta_graph_def.graph_def()));

  // Run the init op. This also tests the case of no output tensors.
  RunMetadata run_metadata;
  TF_ASSERT_OK(session_->Run(
      /*run_options=*/{}, /*inputs=*/{}, /*output_tensor_names=*/{},
      /*target_tensor_names=*/{"init"}, nullptr, &run_metadata));

  const DeviceMgr* device_manager_final;
  TF_ASSERT_OK(session_->LocalDeviceManager(&device_manager_final));
  ASSERT_EQ(device_manager, device_manager_final);
}

TEST_F(TfrtSessionTest, IntraOpThreadPoolOptionWarning) {
  // Create a TfrtSession.
  SessionOptions options;
  options.config.mutable_experimental()->set_use_tfrt(true);
  options.config.set_intra_op_parallelism_threads(1);
  session_.reset(NewSession(options));
  // A session should be created without any errors, though the option is
  // ignored.
  ASSERT_TRUE(session_ != nullptr);
}

TEST_F(TfrtSessionTest, Callable) {
  CallableOptions callable_options;
  std::vector<Tensor> feed_tensors;
  for (auto& input : inputs_) {
    callable_options.add_feed(input.first);
    feed_tensors.emplace_back(input.second);
  }
  for (auto& output : output_tensor_names_) {
    callable_options.add_fetch(output);
  }
  for (auto& target : target_node_names_) {
    callable_options.add_target(target);
  }

  Session::CallableHandle callable_handle;
  TF_ASSERT_OK(session_->MakeCallable(callable_options, &callable_handle));

  std::vector<Tensor> outputs;
  RunMetadata run_metadata;
  TF_ASSERT_OK(session_->RunCallable(callable_handle, feed_tensors, &outputs,
                                     &run_metadata));

  ASSERT_EQ(outputs.size(), 3);

  // Check output "r1".
  test::ExpectEqual(outputs[0],
                    test::AsTensor<int32_t>({6}, TensorShape{1, 1}));
  TF_ASSERT_OK(session_->ReleaseCallable(callable_handle));
}

TEST_F(TfrtSessionTest, Finalize) { TF_ASSERT_OK(session_->Finalize()); }

TEST_F(TfrtSessionTest, WithTargetNodes) {
  std::vector<Tensor> outputs;
  TF_ASSERT_OK(session_->Run(inputs_, output_tensor_names_, target_node_names_,
                             &outputs));

  ASSERT_EQ(outputs.size(), 3);

  // Check output "r1".
  test::ExpectEqual(outputs[0],
                    test::AsTensor<int32_t>({6}, TensorShape{1, 1}));

  // Check output "r21".
  test::ExpectEqual(outputs[1],
                    test::AsTensor<int32_t>({12}, TensorShape{1, 1}));

  // Check output "r31".
  test::ExpectEqual(outputs[2],
                    test::AsTensor<int32_t>({18}, TensorShape{1, 1}));
}

TEST_F(TfrtSessionTest, CreateWithEmptyGraphIsNoop) {
  // Reset the session.
  SessionOptions options;
  options.config.mutable_experimental()->set_use_tfrt(true);
  session_.reset(NewSession(options));
  ASSERT_TRUE(session_ != nullptr);

  // Create the session with an empty GraphDef.
  TF_ASSERT_OK(session_->Create(GraphDef()));

  // Create agian with an unempty GraphDef.
  std::string saved_model_dir = GetDataDependencyFilepath(
      "machina/core/tfrt/saved_model/tests/toy_v1/1");

  MetaGraphDef meta_graph_def;
  TF_ASSERT_OK(ReadMetaGraphDefFromSavedModel(saved_model_dir, {"serve"},
                                              &meta_graph_def));

  TF_ASSERT_OK(session_->Create(meta_graph_def.graph_def()));
}

TEST_F(TfrtSessionTest, CreateAgainError) {
  // On a created session, create agian with a GraphDef.
  std::string saved_model_dir = GetDataDependencyFilepath(
      "machina/core/tfrt/saved_model/tests/toy_v1/1");

  MetaGraphDef meta_graph_def;
  TF_ASSERT_OK(ReadMetaGraphDefFromSavedModel(saved_model_dir, {"serve"},
                                              &meta_graph_def));

  auto status = session_->Create(meta_graph_def.graph_def());

  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(),
              ::testing::HasSubstr(
                  "A Graph has already been created for this session."));
}

TEST_F(TfrtSessionTest, CreateAfterCloseError) {
  // Reset the session.
  SessionOptions options;
  options.config.mutable_experimental()->set_use_tfrt(true);
  session_.reset(NewSession(options));
  ASSERT_TRUE(session_ != nullptr);

  // Close the session.
  TF_ASSERT_OK(session_->Close());

  // Create the session with a GraphDef.
  std::string saved_model_dir = GetDataDependencyFilepath(
      "machina/core/tfrt/saved_model/tests/toy_v1/1");

  MetaGraphDef meta_graph_def;
  TF_ASSERT_OK(ReadMetaGraphDefFromSavedModel(saved_model_dir, {"serve"},
                                              &meta_graph_def));

  auto status = session_->Create(meta_graph_def.graph_def());

  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(),
              ::testing::HasSubstr("Session has been closed."));
}

TEST_F(TfrtSessionTest, ExtendWhenNotCreated) {
  // Reset the session.
  SessionOptions options;
  options.config.mutable_experimental()->set_use_tfrt(true);
  session_.reset(NewSession(options));
  ASSERT_TRUE(session_ != nullptr);

  // Extend the session with a GraphDef.
  std::string saved_model_dir = GetDataDependencyFilepath(
      "machina/core/tfrt/saved_model/tests/toy_v1/1");

  MetaGraphDef meta_graph_def;
  TF_ASSERT_OK(ReadMetaGraphDefFromSavedModel(saved_model_dir, {"serve"},
                                              &meta_graph_def));

  TF_ASSERT_OK(session_->Extend(meta_graph_def.graph_def()));

  // Run the init op. This also tests the case of no output tensors.
  TF_ASSERT_OK(session_->Run(/*inputs=*/{}, /*output_tensor_names=*/{},
                             /*target_tensor_names=*/{"init"}, nullptr));

  // Test run.
  std::vector<Tensor> outputs;
  TF_ASSERT_OK(session_->Run(inputs_, output_tensor_names_,
                             /*target_tensor_names=*/{}, &outputs));

  ASSERT_EQ(outputs.size(), 3);

  // Check output "r1".
  test::ExpectEqual(outputs[0],
                    test::AsTensor<int32_t>({6}, TensorShape{1, 1}));

  // Check output "r21".
  test::ExpectEqual(outputs[1],
                    test::AsTensor<int32_t>({12}, TensorShape{1, 1}));

  // Check output "r31".
  test::ExpectEqual(outputs[2],
                    test::AsTensor<int32_t>({18}, TensorShape{1, 1}));
}

TEST_F(TfrtSessionTest, ExtendAfterCreate) {
  // Reset the session.
  SessionOptions options;
  options.config.mutable_experimental()->set_use_tfrt(true);
  options.config.mutable_experimental()->set_disable_optimize_for_static_graph(
      true);
  session_.reset(NewSession(options));
  ASSERT_TRUE(session_ != nullptr);

  // Create the session with a GraphDef.
  GraphDef graph_def;
  {
    auto scope = machina::Scope::NewRootScope().WithDevice("/device:CPU:0");

    Output a = ops::Const(scope.WithOpName("a"), 0.0f, {10, 10});

    Output b = ops::Const(scope.WithControlDependencies(a).WithOpName("b"),
                          0.0f, {10, 10});
    Output c = ops::Identity(scope.WithOpName("c"), b);

    TF_ASSERT_OK(scope.ToGraphDef(&graph_def));
  }

  TF_ASSERT_OK(session_->Create(graph_def));

  // On the created session, extend with a GraphDef.
  GraphDef extension;
  {
    auto scope = machina::Scope::NewRootScope().WithDevice("/device:CPU:0");

    auto input = ops::Placeholder(scope.WithOpName("input"), DT_INT32);
    auto rank = ops::Rank(scope.WithOpName("rank"), input);

    TF_ASSERT_OK(scope.ToGraphDef(&extension));
  }

  TF_ASSERT_OK(session_->Extend(extension));

  // Set input 'x' to [[1, 1, 1]]
  std::vector<std::pair<std::string, machina::Tensor>> inputs;
  inputs.push_back({"input", machina::tfrt_stub::CreateTfTensor<int32_t>(
                                 /*shape=*/{1, 3}, /*data=*/{1, 1, 1})});

  std::vector<machina::Tensor> outputs;

  TF_ASSERT_OK(session_->Run(inputs,
                             /*output_tensor_names=*/{"rank"},
                             /*target_tensor_names=*/{}, &outputs));
  ASSERT_EQ(outputs.size(), 1);

  EXPECT_THAT(machina::tfrt_stub::GetTfTensorData<int32_t>(outputs[0]),
              ::testing::ElementsAreArray({2}));
}

TEST_F(TfrtSessionTest, ExtendAfterCreate_ErrorWithStaticGraphOptimization) {
  // Reset the session.
  SessionOptions options;
  options.config.mutable_experimental()->set_use_tfrt(true);
  options.config.mutable_experimental()->set_optimize_for_static_graph(true);
  session_.reset(NewSession(options));
  ASSERT_TRUE(session_ != nullptr);

  // Create the session with a GraphDef.
  GraphDef graph_def;
  {
    auto scope = machina::Scope::NewRootScope().WithDevice("/device:CPU:0");

    Output a = ops::Const(scope.WithOpName("a"), 0.0f, {10, 10});

    Output b = ops::Const(scope.WithControlDependencies(a).WithOpName("b"),
                          0.0f, {10, 10});
    Output c = ops::Identity(scope.WithOpName("c"), b);

    TF_ASSERT_OK(scope.ToGraphDef(&graph_def));
  }

  TF_ASSERT_OK(session_->Create(graph_def));

  // On the created session, extend with a GraphDef.
  GraphDef extension;
  {
    auto scope = machina::Scope::NewRootScope().WithDevice("/device:CPU:0");

    auto input = ops::Placeholder(scope.WithOpName("input"), DT_INT32);
    auto rank = ops::Rank(scope.WithOpName("rank"), input);

    TF_ASSERT_OK(scope.ToGraphDef(&extension));
  }

  auto status = session_->Extend(extension);

  ASSERT_FALSE(status.ok());
  EXPECT_THAT(
      status.ToString(),
      ::testing::HasSubstr("Extending the graph is not supported when"));
}

TEST_F(TfrtSessionTest, ExtendAfterCloseError) {
  // Close the session.
  TF_ASSERT_OK(session_->Close());

  // Extend the session with a GraphDef.
  std::string saved_model_dir = GetDataDependencyFilepath(
      "machina/core/tfrt/saved_model/tests/toy_v1/1");

  MetaGraphDef meta_graph_def;
  TF_ASSERT_OK(ReadMetaGraphDefFromSavedModel(saved_model_dir, {"serve"},
                                              &meta_graph_def));

  auto status = session_->Extend(meta_graph_def.graph_def());

  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(),
              ::testing::HasSubstr("Session has been closed."));
}

TEST_F(TfrtSessionTest, RunAfterCloseError) {
  // Close the session.
  TF_ASSERT_OK(session_->Close());

  std::vector<Tensor> outputs;
  auto status = session_->Run(inputs_, output_tensor_names_,
                              /*target_tensor_names=*/{}, &outputs);

  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(),
              ::testing::HasSubstr("Session has been closed."));
}

TEST_F(TfrtSessionTest, InitializeTwiceCrashes) {
  TfrtSessionOptions options;
  auto second_initialize = [](TfrtSessionOptions options) {
    auto status = InitializeTfrtSession(options);
    TF_ASSERT_OK(status);  // Crashes before getting here.
  };
  ASSERT_DEBUG_DEATH(second_initialize(options), "");
}

TEST_F(TfrtSessionTest, GetRuntime) {
  auto runtime = TfrtSessionFactory::GetRuntime();
  EXPECT_NE(runtime, nullptr);
}

TEST_F(TfrtSessionTest, RegisterTwiceCrashes) {
  TfrtSessionFactory::RegisterInitializer(
      [](tfrt_stub::Runtime*) { return absl::OkStatus(); });
  ASSERT_DEBUG_DEATH(TfrtSessionFactory::RegisterInitializer(
                         [](tfrt_stub::Runtime*) { return absl::OkStatus(); }),
                     "");
}
}  // namespace
}  // namespace machina

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

  testing::AddGlobalTestEnvironment(new machina::TfrtSessionEnvironment());

  return RUN_ALL_TESTS();
}
