/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, March 6, 2025.
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

#include "machina/core/util/matmul_autotune.h"

#include "machina/core/framework/types.h"
#include "machina/core/lib/core/stringpiece.h"
#include "machina/core/util/env_var.h"

namespace machina {
bool MatmulAutotuneEnable() {
  bool value;
  absl::Status status =
      ReadBoolFromEnvVar("TF_MATMUL_AUTOTUNE_ENABLE", false, &value);
  if (!status.ok()) {
    LOG(ERROR) << status.message();
  }
  return value;
}

bool MatmulDoFP32ComputationFP16Input() {
  bool value;
  // Feedback from NVIDIA: the "true floating point 16" compute capability is
  // absent from compute capability SM 5.2. The native 16 bit floating point
  // computation was introduced in SM 5.3 and higher compute capability. So
  // for compatibility, set this to be true by default for now.
  // TODO(yangzihao): In the future, we need to return three possibilities:
  // user-set-true, user-set-false, user-no-setting. In the calling sites,
  // check the compatibilities. Note that user-set-false with compute
  // capability <= 5.2 will cause an error in the later cublasGemmEx() call.
  absl::Status status =
      ReadBoolFromEnvVar("TF_FP16_MATMUL_USE_FP32_COMPUTE", true, &value);
  if (!status.ok()) {
    LOG(ERROR) << status.message();
  }
  return value;
}

}  // namespace machina
