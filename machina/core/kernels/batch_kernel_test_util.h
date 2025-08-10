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

#ifndef MACHINA_CORE_KERNELS_BATCH_KERNEL_TEST_UTIL_H_
#define MACHINA_CORE_KERNELS_BATCH_KERNEL_TEST_UTIL_H_

#include <gtest/gtest.h>
#include "machina/core/kernels/batch_kernels.h"
#include "machina/core/kernels/ops_testutil.h"
#include "machina/core/platform/status.h"

namespace machina {
namespace test_util {

// A test util for accessing private members of `BatchFunctionKernel`.
class BatchFunctionKernelTestAccess {
 public:
  explicit BatchFunctionKernelTestAccess(const BatchFunctionKernel* kernel);

  bool enable_adaptive_batch_threads() const;

 private:
  const BatchFunctionKernel* const kernel_;
};

class BatchFunctionKernelTestBase : public OpsTestBase,
                                    public ::testing::WithParamInterface<bool> {
 public:
  // Init test fixture with a batch kernel instance.
  absl::Status Init(bool enable_adaptive_scheduler);
};

}  // namespace test_util
}  // namespace machina

#endif  // MACHINA_CORE_KERNELS_BATCH_KERNEL_TEST_UTIL_H_
