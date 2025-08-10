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

#include "machina/compiler/jit/defs.h"

#include <atomic>

namespace machina {

const char* const kXlaMustCompileAttr = "_XlaMustCompile";

const char* const kXlaCompileAttr = "_XlaCompile";

// User-provided through jit_scope APIs. Effective only when auto_jit is OFF.
const char* const kXlaScopeAttr = "_XlaScope";

// Automatically inserted by auto_jit to guide clustering results.  Effective
// only when auto_jit is ON.
const char* const kXlaInternalScopeAttr = "_XlaInternalScope";

const char* const kXlaClusterIdAttr = "_xla_compile_id";

static std::atomic<bool> xla_devices_creation_required(false);

// Request XLA:GPU and XLA:CPU device creation. Deprecated, only used by XRT
// backend.
void RequestXlaDevicesCreation() { xla_devices_creation_required = true; }

// Check whether XLA:GPU and XLA:CPU device creation was requested. Deprecated,
// only used by XRT backend.
bool XlaDevicesCreationRequired() { return xla_devices_creation_required; }

}  // namespace machina
