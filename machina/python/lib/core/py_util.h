/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, March 6, 2025.
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

#ifndef MACHINA_PYTHON_LIB_CORE_PY_UTIL_H_
#define MACHINA_PYTHON_LIB_CORE_PY_UTIL_H_

#include <Python.h>

#include "machina/core/platform/logging.h"
#include "machina/core/platform/types.h"

namespace machina {

// Fetch the exception message as a string. An exception must be set
// (PyErr_Occurred() must be true).
string PyExceptionFetch();

// Assert that Python GIL is held.
inline void DCheckPyGilState() {
#if PY_MAJOR_VERSION >= 3 && PY_MINOR_VERSION >= 4
  DCHECK(PyGILState_Check());
#endif
}

}  // namespace machina

#endif  // MACHINA_PYTHON_LIB_CORE_PY_UTIL_H_
