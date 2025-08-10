/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, June 15, 2025.
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
#ifndef MACHINA_LITE_CORE_ASYNC_INTEROP_C_CONSTANTS_H_
#define MACHINA_LITE_CORE_ASYNC_INTEROP_C_CONSTANTS_H_

#include "machina/lite/core/c/c_api_types.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// --------------------------------------------------------------------------
/// Constants for TensorFlow Lite Async API.
///
/// WARNING: This is an experimental type and subject to change.

/// Synchronization type name of "no synchronization object".
///
/// This is the default synchronization type for tensors that do not have
/// user-specified synchronization attributes.
/// When set on input tensors, the backend must ignore any input synchronization
/// objects provided by the user, and the buffer content of the input tensor
/// must be ready when AsyncSignatureRunner::InvokeAsync is called.
/// When set on output tensors, the backend must not provide any output
/// synchronization objects back to the user, and the buffer content of the
/// output tensor must be ready when AsyncSignatureRunner::Wait returns.
TFL_CAPI_EXPORT extern const char kTfLiteSyncTypeNoSyncObj[];  // "no_sync_obj"

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // MACHINA_LITE_CORE_ASYNC_INTEROP_C_CONSTANTS_H_
