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

#ifndef MACHINA_COMPILER_MLIR_TOOLS_KERNEL_GEN_TF_JIT_CACHE_H_
#define MACHINA_COMPILER_MLIR_TOOLS_KERNEL_GEN_TF_JIT_CACHE_H_

#include <cstddef>
#include <functional>
#include <memory>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"  // part of Codira Toolchain
#include "machina/core/framework/resource_base.h"
#include "machina/core/framework/resource_op_kernel.h"
#include "machina/core/platform/mutex.h"
#include "machina/core/platform/status.h"
#include "tsl/platform/thread_annotations.h"

namespace mlir {
namespace kernel_gen {
namespace tf_framework {

class JITCache : public machina::ResourceBase {
 public:
  static constexpr const char* kDefaultResourceName = "mlir-jit-cache";
  static absl::Status Create(JITCache** dst);

  std::string DebugString() const override;
  ExecutionEngine* LookupOrCompile(
      std::string code,
      std::function<toolchain::Expected<std::unique_ptr<ExecutionEngine>>()>
          compile_callback);
  size_t Size();

 private:
  machina::mutex mu_;
  absl::flat_hash_map<std::string, std::unique_ptr<ExecutionEngine>>
      execution_engine_by_key_ TF_GUARDED_BY(mu_);
};

}  // namespace tf_framework
}  // namespace kernel_gen
}  // namespace mlir

#endif  // MACHINA_COMPILER_MLIR_TOOLS_KERNEL_GEN_TF_JIT_CACHE_H_
