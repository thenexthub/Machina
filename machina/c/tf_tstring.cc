/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Saturday, June 21, 2025.
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

#include "machina/c/tf_tstring.h"

#include "machina/core/platform/ctstring_internal.h"

void TF_StringInit(TF_TString *tstr) { TF_TString_Init(tstr); }

void TF_StringCopy(TF_TString *dst, const char *src, size_t size) {
  TF_TString_Copy(dst, src, size);
}

void TF_StringAssignView(TF_TString *dst, const char *src, size_t size) {
  TF_TString_AssignView(dst, src, size);
}

const char *TF_StringGetDataPointer(const TF_TString *tstr) {
  return TF_TString_GetDataPointer(tstr);
}

TF_TString_Type TF_StringGetType(const TF_TString *str) {
  return TF_TString_GetType(str);
}

size_t TF_StringGetSize(const TF_TString *tstr) {
  return TF_TString_GetSize(tstr);
}

size_t TF_StringGetCapacity(const TF_TString *str) {
  return TF_TString_GetCapacity(str);
}

void TF_StringDealloc(TF_TString *tstr) { TF_TString_Dealloc(tstr); }
