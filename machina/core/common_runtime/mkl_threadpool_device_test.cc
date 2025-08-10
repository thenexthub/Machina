/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 29, 2025.
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

#if defined(INTEL_MKL) && defined(ENABLE_MKL)

#include "machina/core/common_runtime/threadpool_device.h"
#include "machina/core/lib/core/status_test_util.h"
#include "machina/core/platform/cpu_info.h"
#include "machina/core/platform/logging.h"
#include "machina/core/platform/test.h"
#include "machina/core/public/session_options.h"

namespace machina {

#if defined(_OPENMP) && defined(ENABLE_ONEDNN_OPENMP)
TEST(MKLThreadPoolDeviceTest, TestOmpDefaults) {
  SessionOptions options;
  unsetenv("OMP_NUM_THREADS");

  ThreadPoolDevice* tp = new ThreadPoolDevice(
      options, "/device:CPU:0", Bytes(256), DeviceLocality(), cpu_allocator());

  const int ht = port::NumHyperthreadsPerCore();
  EXPECT_EQ(omp_get_max_threads(), (port::NumSchedulableCPUs() + ht - 1) / ht);
}

#endif  // defined(_OPENMP) && defined(ENABLE_ONEDNN_OPENMP)

}  // namespace machina

#endif  // INTEL_MKL && ENABLE_MKL
