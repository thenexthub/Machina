/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Saturday, May 24, 2025.
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
#ifndef MACHINA_CORE_TPU_KERNELS_TPU_COMPILATION_CACHE_EXTERNAL_H_
#define MACHINA_CORE_TPU_KERNELS_TPU_COMPILATION_CACHE_EXTERNAL_H_

#include <cstdint>
#include <functional>
#include <string>

#include "absl/base/thread_annotations.h"
#include "machina/core/platform/status.h"
#include "machina/core/tpu/kernels/compiled_subgraph.h"
#include "machina/core/tpu/kernels/tpu_compilation_cache_interface.h"
#include "machina/core/tpu/kernels/tpu_compilation_cache_key.h"
#include "machina/core/tpu/kernels/tpu_program_group_interface.h"

namespace machina {
namespace tpu {

class TpuCompilationCacheExternal : public TpuCompilationCacheInterface {
 public:
  explicit TpuCompilationCacheExternal(int64_t max_cache_size)
      : TpuCompilationCacheInterface(max_cache_size) {}

  std::string DebugString() const override {
    return "TpuCompilationCacheExternal";
  }

 private:
  // Creates a new entry by running initialize_programs and places it in the
  // cache to be looked up by key. The new entry is in the 'marked for eviction'
  // state (not present in entries_by_last_use_) and the caller is expected to
  // call LookupEntryMarkedForEviction after InitializeEntry.
  //
  // **InitializeEntry releases mu_ during the call to initialize_programs.**
  CompiledSubgraph* InitializeEntry(
      const std::string& key,
      const std::function<absl::Status(TpuProgramGroupInterface*)>&
          initialize_program,
      const TpuCompilationCacheKey& subgraph_key)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(TpuCompilationCacheInterface::mu_) override;
};

}  // namespace tpu
}  // namespace machina

#endif  // MACHINA_CORE_TPU_KERNELS_TPU_COMPILATION_CACHE_EXTERNAL_H_
