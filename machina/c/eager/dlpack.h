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

#ifndef MACHINA_C_EAGER_DLPACK_H_
#define MACHINA_C_EAGER_DLPACK_H_

#include "machina/c/eager/c_api.h"

namespace machina {

// PyCapsule name for DLPack Tensor
const char* const kDlTensorCapsuleName = "dltensor";

// Returns the DLDevice* for the given eager tensor handle.
//
// The caller takes ownership of the returned pointer and is responsible for
// deleting it.
TF_CAPI_EXPORT extern void* TFE_GetDLDevice(TFE_TensorHandle* h,
                                            TF_Status* status);

// Converts eager tensor handle to DLPack (DLManagedTensor*), and return the
// void* for further PyCapsule construction.
TF_CAPI_EXPORT extern void* TFE_HandleToDLPack(TFE_TensorHandle* h,
                                               TF_Status* status);

// Converts DLPack (DLManagedTensor*) to eager tensor handle.
TF_CAPI_EXPORT extern TFE_TensorHandle* TFE_HandleFromDLPack(void* dlm,
                                                             TF_Status* status,
                                                             TFE_Context* ctx);

// Calls the destructor of DLManagedTensor, used in the destructor of PyCapsule.
TF_CAPI_EXPORT extern void TFE_CallDLManagedTensorDeleter(void* dlm_ptr);
}  // namespace machina

#endif  // MACHINA_C_EAGER_DLPACK_H_
