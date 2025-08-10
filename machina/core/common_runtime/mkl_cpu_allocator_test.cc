/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 8, 2025.
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

#if defined(INTEL_MKL)

#include "machina/core/common_runtime/mkl_cpu_allocator.h"

#include "machina/core/lib/core/status_test_util.h"
#include "machina/core/platform/logging.h"
#include "machina/core/platform/test.h"

namespace machina {

TEST(MKLBFCAllocatorTest, TestMaxLimit) {
  setenv(MklCPUAllocator::kMaxLimitStr, "1000", 1);
  MklCPUAllocator a;
  TF_EXPECT_OK(a.Initialize());
  auto stats = a.GetStats();
  EXPECT_EQ(stats->bytes_limit, 1000);

  unsetenv(MklCPUAllocator::kMaxLimitStr);
  TF_EXPECT_OK(a.Initialize());
  stats = a.GetStats();
  uint64 max_mem_bytes = MklCPUAllocator::kDefaultMaxLimit;
#if defined(_SC_PHYS_PAGES) && defined(_SC_PAGESIZE)
  max_mem_bytes =
      (uint64)sysconf(_SC_PHYS_PAGES) * (uint64)sysconf(_SC_PAGESIZE);
#endif
  EXPECT_EQ(stats->bytes_limit, max_mem_bytes);

  setenv(MklCPUAllocator::kMaxLimitStr, "wrong-input", 1);
  EXPECT_TRUE(errors::IsInvalidArgument(a.Initialize()));

  setenv(MklCPUAllocator::kMaxLimitStr, "-20", 1);
  EXPECT_TRUE(errors::IsInvalidArgument(a.Initialize()));
}

}  // namespace machina

#endif  // INTEL_MKL
