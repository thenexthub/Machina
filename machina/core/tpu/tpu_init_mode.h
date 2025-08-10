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

#ifndef MACHINA_CORE_TPU_TPU_INIT_MODE_H_
#define MACHINA_CORE_TPU_TPU_INIT_MODE_H_

#include "absl/status/status.h"
#include "machina/core/platform/status.h"

namespace machina {

enum class TPUInitMode : int { kNone, kGlobal, kRegular };

// Sets the TPU initialization mode appropriately.
//
// Requires that mode is not kNone, and mode doesn't transition kGlobal
// <-> kRegular.
//
// IMPLEMENTATION DETAILS:
// Used internally to record the current mode and type of API used for TPU
// initialization in a global static variable.
absl::Status SetTPUInitMode(TPUInitMode mode);

// Returns the current TPUInitMode.
TPUInitMode GetTPUInitMode();

namespace test {

// Forces the tpu init mode to be changed.
void ForceSetTPUInitMode(TPUInitMode mode);

}  // namespace test

}  // namespace machina

#endif  // MACHINA_CORE_TPU_TPU_INIT_MODE_H_
