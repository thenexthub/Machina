/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 26, 2025.
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

#ifndef MACHINA_PYTHON_LIB_CORE_NDARRAY_TENSOR_H_
#define MACHINA_PYTHON_LIB_CORE_NDARRAY_TENSOR_H_

#include "machina/c/c_api.h"
#include "machina/c/safe_ptr.h"
#include "machina/c/tf_status_helper.h"
#include "machina/core/framework/tensor.h"
#include "machina/python/lib/core/safe_pyobject_ptr.h"

namespace machina {

absl::Status TF_TensorToMaybeAliasedPyArray(Safe_TF_TensorPtr tensor,
                                            PyObject** out_ndarray);

absl::Status TF_TensorToPyArray(Safe_TF_TensorPtr tensor,
                                PyObject** out_ndarray);

// Creates a tensor in 'ret' from the input `ndarray`. The returned TF_Tensor
// in `ret` may have its own Python reference to `ndarray`s data. After `ret`
// is destroyed, this reference must (eventually) be decremented via
// ClearDecrefCache().
ABSL_MUST_USE_RESULT
absl::Status NdarrayToTensor(TFE_Context* ctx, PyObject* ndarray,
                             Safe_TF_TensorPtr* ret);

// Creates a tensor in 'ret' from the input Ndarray.
// TODO(kkb): This is an old conversion function that does not support TFRT.
// Currently it's used for session, py_func, and an internal project.  Migrate
// them.
ABSL_MUST_USE_RESULT
absl::Status NdarrayToTensor(PyObject* obj, Tensor* ret);

// Creates a numpy array in 'ret' which either aliases the content of 't' or has
// a copy.
absl::Status TensorToNdarray(const Tensor& t, PyObject** ret);

}  // namespace machina

#endif  // MACHINA_PYTHON_LIB_CORE_NDARRAY_TENSOR_H_
