/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, May 15, 2025.
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

#ifndef LIBTPU_EXCLUDE_C_API_IMPL

#include "machina/c/tf_status.h"

#include "machina/c/tf_status_internal.h"

// Trampoline implementation to redirect to TSL. Kept here for backward
// compatibility only.

TF_Status* TF_NewStatus() { return TSL_NewStatus(); }
void TF_DeleteStatus(TF_Status* s) { TSL_DeleteStatus(s); }
void TF_SetStatus(TF_Status* s, TF_Code code, const char* msg) {
  TSL_SetStatus(s, TSL_Code(code), msg);
}
void TF_SetPayload(TF_Status* s, const char* key, const char* value) {
  TSL_SetPayload(s, key, value);
}
void TF_ForEachPayload(const TF_Status* s, TF_PayloadVisitor visitor,
                       void* capture) {
  TSL_ForEachPayload(s, visitor, capture);
}
void TF_SetStatusFromIOError(TF_Status* s, int error_code,
                             const char* context) {
  TSL_SetStatusFromIOError(s, error_code, context);
}
TF_Code TF_GetCode(const TF_Status* s) { return TF_Code(TSL_GetCode(s)); }
const char* TF_Message(const TF_Status* s) { return TSL_Message(s); }

#endif  // LIBTPU_EXCLUDE_C_API_IMPL
