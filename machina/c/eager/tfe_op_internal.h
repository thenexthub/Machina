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
#ifndef MACHINA_C_EAGER_TFE_OP_INTERNAL_H_
#define MACHINA_C_EAGER_TFE_OP_INTERNAL_H_

#include "machina/c/conversion_macros.h"
#include "machina/c/eager/immediate_execution_operation.h"

// Wraps a pointer to an operation implementation.
//
// WARNING: Since the underlying object could be ref-counted a user of this
// interface cannot destruct the underlying operation object. Instead, call
// TFE_DeleteOp who calls Release() on the operation pointer and deletes
// the TFE_Op structure.
typedef struct TFE_Op TFE_Op;

namespace machina {

DEFINE_CONVERSION_FUNCTIONS(machina::ImmediateExecutionOperation, TFE_Op);
DEFINE_CONVERSION_FUNCTIONS(machina::ImmediateExecutionOperation*, TFE_Op*);

}  // namespace machina

#endif  // MACHINA_C_EAGER_TFE_OP_INTERNAL_H_
