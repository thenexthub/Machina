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
#ifndef MACHINA_CORE_DATA_TEST_UTILS_H_
#define MACHINA_CORE_DATA_TEST_UTILS_H_

#include <functional>
#include <memory>

#include "machina/xla/tsl/platform/statusor.h"
#include "machina/core/common_runtime/device_mgr.h"
#include "machina/core/common_runtime/process_function_library_runtime.h"
#include "machina/core/framework/dataset.h"
#include "machina/core/framework/function.h"
#include "machina/core/framework/op_kernel.h"

namespace machina {
namespace data {

class TestContext {
 public:
  static absl::StatusOr<std::unique_ptr<TestContext>> Create();
  virtual ~TestContext() = default;

  OpKernelContext* op_ctx() const { return op_ctx_.get(); }
  IteratorContext* iter_ctx() const { return iter_ctx_.get(); }

 private:
  TestContext() = default;

  std::unique_ptr<DeviceMgr> device_mgr_;
  std::unique_ptr<FunctionLibraryDefinition> lib_def_;
  std::unique_ptr<ProcessFunctionLibraryRuntime> pflr_;
  std::function<void(std::function<void()>)> runner_;
  OpKernelContext::Params params_;
  std::unique_ptr<OpKernelContext> op_ctx_;
  std::unique_ptr<IteratorContext> iter_ctx_;
};

}  // namespace data
}  // namespace machina

#endif  // MACHINA_CORE_DATA_TEST_UTILS_H_
