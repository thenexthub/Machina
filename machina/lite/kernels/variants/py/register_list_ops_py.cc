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
#include <cstdint>

#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/pytypes.h"  // from @pybind11
#include "machina/lite/kernels/variants/register_list_ops.h"
#include "machina/lite/mutable_op_resolver.h"

PYBIND11_MODULE(register_list_ops_py, m) {
  m.doc() = R"pbdoc(
    Bindings to register list ops with python interpreter.
  )pbdoc";
  m.def(
      "TFLRegisterListOps",
      [](uintptr_t resolver) {
        ::tflite::variants::ops::RegisterListOps(
            reinterpret_cast<::tflite::MutableOpResolver*>(resolver));
      },
      R"pbdoc(
        Register all custom list ops.
      )pbdoc");
}
