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
#ifndef MACHINA_LITE_CORE_ASYNC_C_TYPES_H_
#define MACHINA_LITE_CORE_ASYNC_C_TYPES_H_

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

/// Opaque type for TfLiteAsyncKernel.
typedef struct TfLiteAsyncKernel TfLiteAsyncKernel;

/// Opaque type for TfLiteExecutionTask.
///
/// See machina/lite/core/async/c/task.h
/// NOTE: TfLiteExecutionTask is NOT thread-safe.
typedef struct TfLiteExecutionTask TfLiteExecutionTask;

/// Enum tag for specifying whether a tensor is the input or output to the
/// model.
typedef enum TfLiteIoType {
  kTfLiteIoTypeUnknown = 0,
  kTfLiteIoTypeInput = 1,
  kTfLiteIoTypeOutput = 2,
} TfLiteIoType;

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // MACHINA_LITE_CORE_ASYNC_C_TYPES_H_
