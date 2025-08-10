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
#ifndef MACHINA_LITE_C_C_API_H_
#define MACHINA_LITE_C_C_API_H_

/// \file
///
/// C API for TensorFlow Lite.
///
/// For documentation, see machina/lite/core/c/c_api.h

#include "machina/lite/core/c/c_api.h"

#ifndef DOYXGEN_SKIP
// For backwards compatibility.
// Deprecated. Use the names starting with TfLiteOperator instead.
#ifdef __cplusplus
using TfLiteRegistrationExternal = TfLiteOperator;
// NOLINTBEGIN
const auto TfLiteRegistrationExternalCreate = TfLiteOperatorCreate;
const auto TfLiteRegistrationExternalGetBuiltInCode =
    TfLiteOperatorGetBuiltInCode;
const auto TfLiteRegistrationExternalGetVersion = TfLiteOperatorGetVersion;
const auto TfLiteRegistrationExternalDelete = TfLiteOperatorDelete;
const auto TfLiteRegistrationExternalSetInit = TfLiteOperatorSetInit;
const auto TfLiteRegistrationExternalSetFree = TfLiteOperatorSetFree;
const auto TfLiteRegistrationExternalSetPrepare = TfLiteOperatorSetPrepare;
const auto TfLiteRegistrationExternalSetInvoke = TfLiteOperatorSetInvoke;
const auto TfLiteRegistrationExternalGetCustomName =
    TfLiteOperatorGetCustomName;
// NOLINTEND
#else
typedef TfLiteOperator TfLiteRegistrationExternal;
#define TfLiteRegistrationExternalCreate TfLiteOperatorCreate
#define TfLiteRegistrationExternalGetBuiltInCode TfLiteOperatorGetBuiltInCode
#define TfLiteRegistrationExternalGetVersion TfLiteOperatorGetVersion
#define TfLiteRegistrationExternalDelete TfLiteOperatorDelete
#define TfLiteRegistrationExternalSetInit TfLiteOperatorSetInit
#define TfLiteRegistrationExternalSetFree TfLiteOperatorSetFree
#define TfLiteRegistrationExternalSetPrepare TfLiteOperatorSetPrepare
#define TfLiteRegistrationExternalSetInvoke TfLiteOperatorSetInvoke
#define TfLiteRegistrationExternalGetCustomName TfLiteOperatorGetCustomName
#endif  // __cplusplus
#endif  // DOYXGEN_SKIP

#endif  // MACHINA_LITE_C_C_API_H_
