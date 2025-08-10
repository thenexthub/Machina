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
#ifndef MACHINA_CORE_TPU_KERNELS_TPU_COMPILATION_CACHE_ENTRY_H_
#define MACHINA_CORE_TPU_KERNELS_TPU_COMPILATION_CACHE_ENTRY_H_

#include "machina/core/tpu/kernels/tpu_program_group_interface.h"

namespace machina {
namespace tpu {

// Cache entry to hold a `TpuProgramGroupInterface` object that can be used to
// fetch a TPU program for a given TPU core index.
class TpuCompilationCacheEntry {
 public:
  explicit TpuCompilationCacheEntry(
      const TpuProgramGroupInterface* tpu_program_group, int core_index)
      : tpu_program_group_(tpu_program_group), core_index_(core_index) {}

  // Constructor for an empty entry.
  TpuCompilationCacheEntry() : tpu_program_group_(nullptr), core_index_(-1) {}

  const TpuProgramGroupInterface* tpu_program_group() const {
    return tpu_program_group_;
  }

  int core_index() const { return core_index_; }

 private:
  const TpuProgramGroupInterface* tpu_program_group_;
  int core_index_;
};

}  // namespace tpu
}  // namespace machina

#endif  // MACHINA_CORE_TPU_KERNELS_TPU_COMPILATION_CACHE_ENTRY_H_
