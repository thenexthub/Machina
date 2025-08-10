/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Friday, April 18, 2025.
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

#ifndef MACHINA_SERVING_CORE_TEST_UTIL_MOCK_SESSION_H_
#define MACHINA_SERVING_CORE_TEST_UTIL_MOCK_SESSION_H_

#include <gmock/gmock.h>
#include "machina/core/platform/threadpool_options.h"
#include "machina/core/public/session.h"

namespace machina {
namespace serving {
namespace test_util {

// A mock of machina::Session.
class MockSession : public machina::Session {
 public:
  MockSession() : Session() {
    ON_CALL(*this, Close()).WillByDefault(::testing::Return(Status()));
  }
  MOCK_METHOD(::machina::Status, Create, (const GraphDef& graph),
              (override));
  MOCK_METHOD(::machina::Status, Extend, (const GraphDef& graph),
              (override));
  MOCK_METHOD(::machina::Status, Run,
              ((const std::vector<std::pair<string, Tensor>>& inputs),
               const std::vector<string>& output_names,
               const std::vector<string>& target_nodes,
               std::vector<Tensor>* outputs),
              (override));
  MOCK_METHOD(::machina::Status, Run,
              (const RunOptions& run_options,
               (const std::vector<std::pair<string, Tensor>>& inputs),
               const std::vector<string>& output_names,
               const std::vector<string>& target_nodes,
               std::vector<Tensor>* outputs, RunMetadata* run_metadata),
              (override));
  MOCK_METHOD(
      ::machina::Status, Run,
      (const RunOptions& run_options,
       (const std::vector<std::pair<string, Tensor>>& inputs),
       const std::vector<string>& output_names,
       const std::vector<string>& target_nodes, std::vector<Tensor>* outputs,
       RunMetadata* run_metadata,
       const machina::thread::ThreadPoolOptions& thread_pool_options),
      (override));
  MOCK_METHOD(::machina::Status, PRunSetup,
              (const std::vector<string>& input_names,
               const std::vector<string>& output_names,
               const std::vector<string>& target_nodes, string* handle),
              (override));
  MOCK_METHOD(::machina::Status, PRun,
              (const string& handle,
               (const std::vector<std::pair<string, Tensor>>& inputs),
               const std::vector<string>& output_names,
               std::vector<Tensor>* outputs),
              (override));

  MOCK_METHOD(::machina::Status, ListDevices,
              (std::vector<::machina::DeviceAttributes> * response),
              (override));

  MOCK_METHOD(::machina::Status, Close, (), (override));
};

}  // namespace test_util
}  // namespace serving
}  // namespace machina

#endif  // MACHINA_SERVING_CORE_TEST_UTIL_MOCK_SESSION_H_
