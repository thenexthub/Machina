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
#include "machina/lite/experimental/acceleration/mini_benchmark/set_big_core_affinity.h"

#include <sched.h>
#include <unistd.h>

#include <cerrno>

#include "machina/lite/experimental/acceleration/mini_benchmark/big_little_affinity.h"

namespace tflite {
namespace acceleration {

int32_t SetBigCoresAffinity() {
#ifdef __ANDROID__
  ::tflite::acceleration::BigLittleAffinity affinity =
      ::tflite::acceleration::GetAffinity();

  cpu_set_t set;
  CPU_ZERO(&set);
  for (int i = 0; i < 16; i++) {
    if (affinity.big_core_affinity & (0x1 << i)) {
      CPU_SET(i, &set);
    }
  }
  if (sched_setaffinity(getpid(), sizeof(set), &set) != -1) {
    return 0;
  } else {
    return errno;
  }
#else  // !__ANDROID__
  return 0;
#endif
}

}  // namespace acceleration
}  // namespace tflite
