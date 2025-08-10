/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, May 15, 2025.
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

#include "machina/lite/delegates/gpu/cl/cl_memory.h"

namespace tflite {
namespace gpu {
namespace cl {

cl_mem_flags ToClMemFlags(AccessType access_type) {
  switch (access_type) {
    case AccessType::READ:
      return CL_MEM_READ_ONLY;
    case AccessType::WRITE:
      return CL_MEM_WRITE_ONLY;
    case AccessType::READ_WRITE:
      return CL_MEM_READ_WRITE;
  }

  return CL_MEM_READ_ONLY;  // unreachable
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
