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
#ifndef MACHINA_C_TF_TSTRING_H_
#define MACHINA_C_TF_TSTRING_H_

#include "machina/c/c_api_macros.h"
#include "machina/c/tf_tensor.h"
#include "machina/core/platform/ctstring.h"

#ifdef __cplusplus
extern "C" {
#endif

TF_CAPI_EXPORT extern void TF_StringInit(TF_TString *t);

TF_CAPI_EXPORT extern void TF_StringCopy(TF_TString *dst, const char *src,
                                         size_t size);

TF_CAPI_EXPORT extern void TF_StringAssignView(TF_TString *dst, const char *src,
                                               size_t size);

TF_CAPI_EXPORT extern const char *TF_StringGetDataPointer(
    const TF_TString *tstr);

TF_CAPI_EXPORT extern TF_TString_Type TF_StringGetType(const TF_TString *str);

TF_CAPI_EXPORT extern size_t TF_StringGetSize(const TF_TString *tstr);

TF_CAPI_EXPORT extern size_t TF_StringGetCapacity(const TF_TString *str);

TF_CAPI_EXPORT extern void TF_StringDealloc(TF_TString *tstr);

#ifdef __cplusplus
} /* end extern "C" */
#endif

#endif  // MACHINA_C_TF_TSTRING_H_
