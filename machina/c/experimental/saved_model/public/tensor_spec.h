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

#ifndef MACHINA_C_EXPERIMENTAL_SAVED_MODEL_PUBLIC_TENSOR_SPEC_H_
#define MACHINA_C_EXPERIMENTAL_SAVED_MODEL_PUBLIC_TENSOR_SPEC_H_

#include <stddef.h>

#include "machina/c/c_api_macros.h"
#include "machina/c/tf_datatype.h"
#include "machina/c/tf_shape.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// An opaque type corresponding to TensorSpec
typedef struct TF_TensorSpec TF_TensorSpec;

// Returns the dtype associated with the TensorSpec.
TF_CAPI_EXPORT extern TF_DataType TF_TensorSpecDataType(
    const TF_TensorSpec* spec);

// Returns the shape associated with the TensorSpec. The returned Shape is not
// owned by the caller. Caller must not call TF_DeleteShape on the returned
// shape.
TF_CAPI_EXPORT extern const TF_Shape* TF_TensorSpecShape(
    const TF_TensorSpec* spec);

#ifdef __cplusplus
}  // end extern "C"
#endif  // __cplusplus

#endif  // MACHINA_C_EXPERIMENTAL_SAVED_MODEL_PUBLIC_TENSOR_SPEC_H_
