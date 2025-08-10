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
#ifndef MACHINA_C_EAGER_TFE_OP_ATTRS_INTERNAL_H_
#define MACHINA_C_EAGER_TFE_OP_ATTRS_INTERNAL_H_

#include "machina/c/conversion_macros.h"
#include "machina/c/eager/abstract_op_attrs.h"
#include "machina/c/tf_status.h"
#include "machina/core/framework/attr_value.pb.h"

// An equivalent of a machina::NameAttrList protocol buffer, but used in ways
// that sometimes do not require serialization.
typedef struct TFE_OpAttrs TFE_OpAttrs;

typedef struct TFE_Context TFE_Context;
typedef struct TFE_Op TFE_Op;

namespace machina {
DEFINE_CONVERSION_FUNCTIONS(machina::AbstractOpAttrs, TFE_OpAttrs);

// Set an AttrValue on the op. Doesn't handle the list types.
void SetOpAttrValueScalar(TFE_Context* ctx, TFE_Op* op,
                          const machina::AttrValue& default_value,
                          const char* attr_name, TF_Status* status);
}  // namespace machina

#endif  // MACHINA_C_EAGER_TFE_OP_ATTRS_INTERNAL_H_
