/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Saturday, June 21, 2025.
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

// Defines the 'constants' wrapper submodule.

#ifndef MACHINA_PYTHON_SAVED_MODEL_PYWRAP_SAVED_MODEL_CONSTANTS_H_
#define MACHINA_PYTHON_SAVED_MODEL_PYWRAP_SAVED_MODEL_CONSTANTS_H_

// placeholder for index annotation headers
#include "pybind11/pybind11.h"  // from @pybind11

namespace machina {
namespace saved_model {
namespace python {

// Wraps the SavedModel constants for exporting to Python.
void DefineConstantsModule(pybind11::module main_module);

}  // namespace python
}  // namespace saved_model
}  // namespace machina

#endif  // MACHINA_PYTHON_SAVED_MODEL_PYWRAP_SAVED_MODEL_CONSTANTS_H_
