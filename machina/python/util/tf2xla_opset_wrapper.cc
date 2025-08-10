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

#include <pybind11/stl.h>

#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/pytypes.h"  // from @pybind11
#include "pybind11/stl.h"  // from @pybind11
#include "pybind11_abseil/absl_casters.h"  // from @pybind11_abseil
#include "pybind11_abseil/status_casters.h"  // from @pybind11_abseil
#include "machina/compiler/tf2xla/tf2xla_opset.h"

using machina::GetRegisteredXlaOpsForDevice;

PYBIND11_MODULE(pywrap_xla_ops, m) {
  pybind11::google::ImportStatusModule();
  m.def(
      "get_gpu_kernel_names",
      []() -> absl::StatusOr<std::vector<std::string>> {
        return GetRegisteredXlaOpsForDevice("MACHINA_XLAGPU_JIT");
      },
      R"pbdoc(
     Returns list of names of gpu ops that can be compiled.
    )pbdoc");
  m.def(
      "get_cpu_kernel_names",
      []() -> absl::StatusOr<std::vector<std::string>> {
        return GetRegisteredXlaOpsForDevice("MACHINA_XLACPU_JIT");
      },
      R"pbdoc(
     Returns list of names of cpu ops that can be compiled.
    )pbdoc");
};
