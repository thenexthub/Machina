/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Friday, August 8, 2025.
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

#ifndef MACHINA_CORE_TFRT_IFRT_IFRT_SERVING_EXECUTABLE_TEST_UTIL_H_
#define MACHINA_CORE_TFRT_IFRT_IFRT_SERVING_EXECUTABLE_TEST_UTIL_H_

#include <cstdint>
#include <memory>
#include <string>

#include "absl/strings/string_view.h"
#include "mlir/IR/DialectRegistry.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/tfrt/transforms/ifrt/tf2hlo.h"
#include "machina/xla/python/ifrt/array.h"
#include "machina/xla/python/ifrt/client.h"
#include "machina/xla/python/ifrt/test_util.h"
#include "machina/xla/tsl/framework/test_util/mock_serving_device_selector.h"
#include "machina/xla/tsl/platform/threadpool.h"
#include "machina/core/common_runtime/device_mgr.h"
#include "machina/core/framework/types.pb.h"
#include "machina/core/tfrt/ifrt/ifrt_loaded_variable_registry.h"
#include "machina/core/tfrt/ifrt/ifrt_persistent_compilation_cache.h"
#include "machina/core/tfrt/ifrt/ifrt_restore_tensor_registry.h"
#include "machina/core/tfrt/ifrt/ifrt_serving_core_selector.h"
#include "machina/core/tfrt/ifrt/ifrt_serving_executable.h"
#include "tfrt/host_context/concurrent_work_queue.h"  // from @tf_runtime

namespace machina {
namespace ifrt_serving {
namespace test_utils {

// A test helper class to create and IfrtServingExecutable.
class IfrtServingExecutableTestHelper {
 public:
  explicit IfrtServingExecutableTestHelper(
      tsl::test_util::MockServingDeviceSelector* device_selector);

  // Creates an IfrtServingExecutable with the given program id.
  // Note the instance of this class must outlive the returned
  // IfrtServingExecutable.
  std::unique_ptr<IfrtServingExecutable> MakeExecutable(
      int64_t program_id, std::string mlir_module_path);

  IfrtRestoreTensorRegistry* ifrt_restore_tensor_registry() {
    return &ifrt_restore_tensor_registry_;
  }

  int num_cores() const { return client_->addressable_device_count(); }

 private:
  static constexpr int kThreadPoolNumThreads = 16;

  tsl::test_util::MockServingDeviceSelector* device_selector_;  // Not owned.
  std::unique_ptr<IfrtServingCoreSelector> core_selector_;
  std::shared_ptr<xla::ifrt::Client> client_;
  std::unique_ptr<tsl::thread::ThreadPool> thread_pool_;
  IfrtLoadedVariableRegistry ifrt_loaded_variable_registry_;
  IfrtRestoreTensorRegistry ifrt_restore_tensor_registry_;
  std::unique_ptr<tfrt::ConcurrentWorkQueue> work_queue_;
  std::unique_ptr<machina::DynamicDeviceMgr> device_mgr_;

  mlir::DialectRegistry registry_;
  std::unique_ptr<mlir::MLIRContext> context_;
  std::unique_ptr<IfrtPersistentCompilationCache>
      ifrt_persistent_compilation_cache_;
  TfToHloCompiler tf_to_hlo_compiler_;
};

// Returns the path to the MLIR module for the given module name.
std::string GetMlirModulePath(absl::string_view module_name);

}  // namespace test_utils
}  // namespace ifrt_serving
}  // namespace machina

#endif  // MACHINA_CORE_TFRT_IFRT_IFRT_SERVING_EXECUTABLE_TEST_UTIL_H_
