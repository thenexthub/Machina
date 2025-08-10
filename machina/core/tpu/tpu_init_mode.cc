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

#include "machina/core/tpu/tpu_init_mode.h"

#include "absl/status/status.h"
#include "machina/core/platform/errors.h"
#include "machina/core/platform/mutex.h"
#include "machina/core/platform/status.h"
#include "tsl/platform/thread_annotations.h"

namespace machina {

namespace {

mutex init_mode_mutex(LINKER_INITIALIZED);
TPUInitMode init_mode TF_GUARDED_BY(init_mode_mutex);

}  // namespace

namespace test {

void ForceSetTPUInitMode(const TPUInitMode mode) {
  mutex_lock l(init_mode_mutex);
  init_mode = mode;
}

}  // namespace test

absl::Status SetTPUInitMode(const TPUInitMode mode) {
  if (mode == TPUInitMode::kNone) {
    return errors::InvalidArgument("State cannot be set to: ",
                                   static_cast<int>(mode));
  }
  {
    mutex_lock l(init_mode_mutex);
    if (init_mode != TPUInitMode::kNone && mode != init_mode) {
      return errors::FailedPrecondition(
          "TPUInit already attempted with mode: ", static_cast<int>(init_mode),
          " and cannot be changed to: ", static_cast<int>(mode),
          ". You are most probably trying to initialize the TPU system, both "
          "using the explicit API and using an initialization Op within the "
          "graph; please choose one. ");
    }
    init_mode = mode;
  }
  return absl::OkStatus();
}

TPUInitMode GetTPUInitMode() {
  mutex_lock l(init_mode_mutex);
  return init_mode;
}

}  // namespace machina
