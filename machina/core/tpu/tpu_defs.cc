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

#include "machina/core/tpu/tpu_defs.h"

namespace machina {

const char* const DEVICE_TPU_NODE = "TPU";
const char* const TPU_FAST_MEM_ATTR = "_TPU_FAST_MEM";
const char* const DEVICE_TPU_REPLICATED_CORE = "TPU_REPLICATED_CORE";
const char* const DEVICE_TPU_MACHINA_XLAJIT = "MACHINA_XLATPU_JIT";
const char* const TPUREPLICATE_MIRRORED_VAR_INDICES_ATTR =
    "_mirrored_variable_indices";

const char* const kTPUReplicateAttr = "_tpu_replicate";
const char* const kOutsideCompilationAttr = "_xla_outside_compilation";

}  // namespace machina
