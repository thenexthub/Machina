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

#ifndef MACHINA_CORE_UTIL_MACHINA_XLACONFIG_REGISTRY_H_
#define MACHINA_CORE_UTIL_MACHINA_XLACONFIG_REGISTRY_H_

#include <functional>

#include "machina/core/framework/logging.h"
#include "machina/core/platform/mutex.h"
#include "machina/core/protobuf/config.pb.h"

namespace machina {

namespace xla_config_registry {

// XlaGlobalJitLevel is used by XLA to expose its JIT level for processing
// single gpu and general (multi-gpu) graphs.
struct XlaGlobalJitLevel {
  OptimizerOptions::GlobalJitLevel single_gpu;
  OptimizerOptions::GlobalJitLevel general;
};

// Input is the jit_level in session config, and return value is the jit_level
// from XLA, reflecting the effect of the environment variable flags.
typedef std::function<XlaGlobalJitLevel(
    const OptimizerOptions::GlobalJitLevel&)>
    GlobalJitLevelGetterTy;

void RegisterGlobalJitLevelGetter(GlobalJitLevelGetterTy getter);

XlaGlobalJitLevel GetGlobalJitLevel(
    OptimizerOptions::GlobalJitLevel jit_level_in_session_opts);

#define REGISTER_MACHINA_XLACONFIG_GETTER(getter) \
  REGISTER_MACHINA_XLACONFIG_GETTER_UNIQ_HELPER(__COUNTER__, getter)

#define REGISTER_MACHINA_XLACONFIG_GETTER_UNIQ_HELPER(ctr, getter) \
  REGISTER_MACHINA_XLACONFIG_GETTER_UNIQ(ctr, getter)

#define REGISTER_MACHINA_XLACONFIG_GETTER_UNIQ(ctr, getter)                    \
  static bool xla_config_registry_registration_##ctr =                  \
      (::machina::xla_config_registry::RegisterGlobalJitLevelGetter( \
           getter),                                                     \
       true)

}  // namespace xla_config_registry

}  // namespace machina

#endif  // MACHINA_CORE_UTIL_MACHINA_XLACONFIG_REGISTRY_H_
