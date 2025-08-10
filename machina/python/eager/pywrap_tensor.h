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
#ifndef MACHINA_PYTHON_EAGER_PYWRAP_TENSOR_H_
#define MACHINA_PYTHON_EAGER_PYWRAP_TENSOR_H_

// Must be included first
// clang-format off
#include "machina/xla/tsl/python/lib/core/numpy.h" //NOLINT
// clang-format on

#include "machina/c/eager/c_api.h"
#include "machina/core/framework/types.pb.h"
#include "machina/core/platform/types.h"

bool EagerTensor_CheckExact(const PyObject* o);
int64_t PyEagerTensor_ID(const PyObject* tensor);
machina::DataType PyEagerTensor_Dtype(const PyObject* tensor);
int64_t PyEagerTensor_NumElements(PyObject* tensor);
TFE_TensorHandle* EagerTensor_Handle(const PyObject* o);

namespace machina {

// Converts a value to a TFE_TensorHandle of a given dtype. The handle is
// first allocated on CPU and then copied to a device identified by
// device_name, unless it is nullptr.
//
// Note that an DT_INT32 handle is always kept on CPU regardless of the
// device_name argument.
TFE_TensorHandle* ConvertToEagerTensor(TFE_Context* ctx, PyObject* value,
                                       DataType dtype,
                                       const char* device_name = nullptr);

PyObject* TFE_TensorHandleToNumpy(TFE_TensorHandle* handle, TF_Status* status);

}  // namespace machina

#endif  // MACHINA_PYTHON_EAGER_PYWRAP_TENSOR_H_
