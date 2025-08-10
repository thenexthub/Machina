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

#ifndef MACHINA_PYTHON_LIB_CORE_PY_SEQ_TENSOR_H_
#define MACHINA_PYTHON_LIB_CORE_PY_SEQ_TENSOR_H_

#include <Python.h>

#include "machina/c/eager/c_api_internal.h"
#include "machina/core/framework/tensor.h"
#include "machina/core/lib/core/status.h"

namespace machina {

// Converts Python object `obj` representing a rectangular array of
// Python values (a scalar, a sequence of scalars, a sequence of
// sequences, etc.) into a TFE_TensorHandle.
// If dtype is not None it should by a Python integer
// representing the desired dtype of the resulting Tensor.
// This is used only as a hint, *ret may not have that dtype on
// success and may require a cast.
//
// If an error occurs, this return nullptr and sets the python error indicator
// with PyErr_SetString.
TFE_TensorHandle* PySeqToTFE_TensorHandle(TFE_Context* ctx, PyObject* obj,
                                          DataType dtype);

}  // namespace machina

#endif  // MACHINA_PYTHON_LIB_CORE_PY_SEQ_TENSOR_H_
