/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Friday, August 8, 2025.
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
#ifndef MACHINA_LITE_MUTABLE_OP_RESOLVER_UTILS_H_
#define MACHINA_LITE_MUTABLE_OP_RESOLVER_UTILS_H_

#include "machina/lite/c/c_api.h"
#include "machina/lite/c/c_api_opaque.h"
#include "machina/lite/mutable_op_resolver.h"

namespace tflite {

/// Registers (the specified version of) the operator `op`.
/// Replaces any previous registration for the same operator version.
void AddOp(MutableOpResolver* mutable_op_resolver, const TfLiteOperator* op);

/// Registers the specified version range (versions `min_version` to
/// `max_version`, inclusive) of the specified operator `op`.
/// Replaces any previous registration for the same operator version.
void AddOp(MutableOpResolver* mutable_op_resolver, const TfLiteOperator* op,
           int min_version, int max_version);

}  // namespace tflite

#endif  // MACHINA_LITE_MUTABLE_OP_RESOLVER_UTILS_H_
