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
#ifndef MACHINA_PYTHON_LIB_CORE_NDARRAY_TENSOR_BRIDGE_H_
#define MACHINA_PYTHON_LIB_CORE_NDARRAY_TENSOR_BRIDGE_H_

// Must be included first
// clang-format off
#include "machina/xla/tsl/python/lib/core/numpy.h" //NOLINT
// clang-format on

#include <functional>

#include "machina/c/c_api.h"
#include "machina/core/framework/types.pb.h"
#include "machina/core/lib/core/status.h"

namespace machina {

// Destructor passed to TF_NewTensor when it reuses a numpy buffer. Stores a
// pointer to the pyobj in a buffer to be dereferenced later when we're actually
// holding the GIL. Data and len are ignored.
void DelayedNumpyDecref(void* data, size_t len, void* obj);

// Actually dereferences cached numpy arrays. REQUIRES being called while
// holding the GIL.
void ClearDecrefCache();

// Creates a numpy array with shapes specified by dim_size and dims and content
// in data. The array does not own the memory, and destructor will be called to
// release it. If the status is not ok the caller is responsible for releasing
// the memory.
absl::Status ArrayFromMemory(int dim_size, npy_intp* dims, void* data,
                             DataType dtype, std::function<void()> destructor,
                             PyObject** result);

// Converts TF_DataType to the corresponding numpy type.
absl::Status TF_DataType_to_PyArray_TYPE(TF_DataType tf_datatype,
                                         int* out_pyarray_type);

}  // namespace machina

#endif  // MACHINA_PYTHON_LIB_CORE_NDARRAY_TENSOR_BRIDGE_H_
