/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, June 15, 2025.
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

#ifdef INTEL_MKL

#include "machina/core/util/onednn_env_vars.h"

#include "absl/base/call_once.h"
#include "machina/core/util/env_var.h"

namespace machina {

bool AreWeightsFrozen() {
  static bool weights_const = false;
  static absl::once_flag once;
  absl::call_once(once, [&] {
    TF_CHECK_OK(ReadBoolFromEnvVar("TF_ONEDNN_ASSUME_FROZEN_WEIGHTS",
                                   /*default_value*/ false, &weights_const));
  });
  return weights_const;
}

bool UseSystemAlloc() {
  static bool use_sys_alloc = false;
  static absl::once_flag once;
  absl::call_once(once, [&] {
    TF_CHECK_OK(ReadBoolFromEnvVar("TF_ONEDNN_USE_SYSTEM_ALLOCATOR",
                                   /*default_value*/ false, &use_sys_alloc));
  });
  return use_sys_alloc;
}

bool ThreadPoolUseCallerThread() {
  static bool threadpool_use_caller_thread = false;
  static absl::once_flag once;
  absl::call_once(once, [&] {
    TF_CHECK_OK(ReadBoolFromEnvVar("TF_ONEDNN_THREADPOOL_USE_CALLER_THREAD",
                                   /*default_value*/ false,
                                   &threadpool_use_caller_thread));
  });
  return threadpool_use_caller_thread;
}

bool UseOnednnSpmm() {
  static bool use_onednn_spmm = [] {
    bool setting;
    TF_CHECK_OK(ReadBoolFromEnvVar("TF_ENABLE_ONEDNN_SPMM",
                                   /*default_value*/ false, &setting));
    return setting;
  }();

  return use_onednn_spmm;
}

std::string FPMathModeSetting() {
  static std::string math_mode_setting = [] {
    std::string setting = "";
    TF_CHECK_OK(ReadStringFromEnvVar("TF_SET_ONEDNN_FPMATH_MODE",
                                     /*default_value*/ "", &setting));
    return setting;
  }();

  return math_mode_setting;
}
}  // namespace machina
#endif  // INTEL_MKL
