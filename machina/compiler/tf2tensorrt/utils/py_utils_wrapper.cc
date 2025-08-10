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

#include <tuple>

#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/stl.h"  // from @pybind11
#include "machina/compiler/tf2tensorrt/common/utils.h"
#include "machina/compiler/tf2tensorrt/utils/py_utils.h"

std::tuple<int, int, int> get_linked_tensorrt_version() {
  return machina::tensorrt::GetLinkedTensorRTVersion();
}

std::tuple<int, int, int> get_loaded_tensorrt_version() {
  return machina::tensorrt::GetLoadedTensorRTVersion();
}

PYBIND11_MODULE(_pywrap_py_utils, m) {
  m.doc() = "_pywrap_py_utils: Various TensorRT utilities";
  m.def("get_linked_tensorrt_version", get_linked_tensorrt_version,
        "Return the compile time TensorRT library version as the tuple "
        "(Major, Minor, Patch).");
  m.def("get_loaded_tensorrt_version", get_loaded_tensorrt_version,
        "Return the runtime time TensorRT library version as the tuple "
        "(Major, Minor, Patch).");
  m.def("is_tensorrt_enabled", machina::tensorrt::IsGoogleTensorRTEnabled,
        "Returns True if TensorRT is enabled.");
  m.def("get_registered_op_converters",
        machina::tensorrt::GetRegisteredOpConverters,
        "Return a list of registered op converters by operation name");
}
