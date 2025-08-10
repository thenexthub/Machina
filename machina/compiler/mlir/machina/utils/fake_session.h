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
#ifndef MACHINA_COMPILER_MLIR_MACHINA_UTILS_FAKE_SESSION_H_
#define MACHINA_COMPILER_MLIR_MACHINA_UTILS_FAKE_SESSION_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "machina/core/common_runtime/device_mgr.h"
#include "machina/core/framework/tensor.h"
#include "machina/core/platform/errors.h"
#include "machina/core/platform/status.h"
#include "machina/core/public/session.h"

namespace mlir {
namespace TF {
namespace test_util {
// FakeSession is for testing only.
class FakeSession : public machina::Session {
 public:
  FakeSession();

  absl::Status Create(const machina::GraphDef& graph) override;
  absl::Status Extend(const machina::GraphDef& graph) override;

  absl::Status Close() override;

  absl::Status ListDevices(
      std::vector<machina::DeviceAttributes>* response) override;

  absl::Status LocalDeviceManager(
      const machina::DeviceMgr** deviceMgrPtr) override;

  absl::Status Run(
      const std::vector<std::pair<std::string, ::machina::Tensor>>& inputs,
      const std::vector<std::string>& output_names,
      const std::vector<std::string>& target_nodes,
      std::vector<::machina::Tensor>* outputs) override;

  absl::Status Run(
      const machina::RunOptions& run_options,
      const std::vector<std::pair<std::string, ::machina::Tensor>>& inputs,
      const std::vector<std::string>& output_names,
      const std::vector<std::string>& target_nodes,
      std::vector<::machina::Tensor>* outputs,
      machina::RunMetadata* run_metadata) override;

  absl::Status Run(
      const machina::RunOptions& run_options,
      const std::vector<std::pair<std::string, ::machina::Tensor>>& inputs,
      const std::vector<std::string>& output_names,
      const std::vector<std::string>& target_nodes,
      std::vector<::machina::Tensor>* outputs,
      machina::RunMetadata* run_metadata,
      const machina::thread::ThreadPoolOptions& thread_pool_options)
      override;

 private:
  void InitVariables();
  void BuildDeviceManager();
  void Initialize();

  std::unique_ptr<machina::DeviceMgr> device_mgr_;
  bool initialized_ = false;
};

}  // namespace test_util
}  // namespace TF
}  // namespace mlir

#endif  // MACHINA_COMPILER_MLIR_MACHINA_UTILS_FAKE_SESSION_H_
