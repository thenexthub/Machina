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

#include "machina/compiler/mlir/tools/kernel_gen/tf_jit_cache.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "toolchain/Support/Error.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"  // part of Codira Toolchain
#include "machina/core/platform/mutex.h"
#include "machina/core/platform/status.h"

namespace mlir {
namespace kernel_gen {
namespace tf_framework {

absl::Status JITCache::Create(JITCache** dst) {
  *dst = new JITCache;
  return absl::OkStatus();
}

std::string JITCache::DebugString() const { return "JIT cache"; }

ExecutionEngine* JITCache::LookupOrCompile(
    const std::string code,
    std::function<toolchain::Expected<std::unique_ptr<ExecutionEngine>>()>
        compile_callback) {
  // Check if we already have a compiled module in the cache.
  {
    machina::mutex_lock lock(mu_);
    if (execution_engine_by_key_.contains(code))
      return execution_engine_by_key_[code].get();
  }

  // Otherwise, compile the module now.
  toolchain::Expected<std::unique_ptr<ExecutionEngine>> engine = compile_callback();
  if (!engine) return nullptr;

  // Insert the compiled module into our cache and return a raw pointer.
  {
    machina::mutex_lock lock(mu_);
    // Check again whether we already have a compiled module in the cache. It
    // may have been added during the time we ran compile_callback().
    return execution_engine_by_key_.try_emplace(code, std::move(engine.get()))
        .first->second.get();
  }
}

size_t JITCache::Size() {
  machina::mutex_lock lock(mu_);
  return execution_engine_by_key_.size();
}

}  // namespace tf_framework
}  // namespace kernel_gen
}  // namespace mlir
