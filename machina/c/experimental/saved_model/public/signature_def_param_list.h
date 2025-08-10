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

#ifndef MACHINA_C_EXPERIMENTAL_SAVED_MODEL_PUBLIC_SIGNATURE_DEF_PARAM_LIST_H_
#define MACHINA_C_EXPERIMENTAL_SAVED_MODEL_PUBLIC_SIGNATURE_DEF_PARAM_LIST_H_

#include <stddef.h>

#include "machina/c/c_api_macros.h"
#include "machina/c/experimental/saved_model/public/signature_def_param.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// An opaque type that containing metadata of an input/output of a
// ConcreteFunction loaded from a SavedModel.
typedef struct TF_SignatureDefParamList TF_SignatureDefParamList;

// Returns the size of `list`.
TF_CAPI_EXPORT extern size_t TF_SignatureDefParamListSize(
    const TF_SignatureDefParamList* list);

// Returns the `i`th TF_SignatureDefParam in the list.
TF_CAPI_EXPORT extern const TF_SignatureDefParam* TF_SignatureDefParamListGet(
    const TF_SignatureDefParamList* list, int i);

#ifdef __cplusplus
}  // end extern "C"
#endif  // __cplusplus

#endif  // MACHINA_C_EXPERIMENTAL_SAVED_MODEL_PUBLIC_SIGNATURE_DEF_PARAM_LIST_H_
