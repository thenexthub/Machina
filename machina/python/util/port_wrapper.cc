/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, May 15, 2025.
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

#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/pytypes.h"  // from @pybind11
#include "machina/core/util/port.h"

PYBIND11_MODULE(_pywrap_util_port, m) {
  m.def("IsGoogleCudaEnabled", machina::IsGoogleCudaEnabled);
  m.def("IsBuiltWithROCm", machina::IsBuiltWithROCm);
  m.def("IsBuiltWithXLA", machina::IsBuiltWithXLA);
  m.def("IsBuiltWithNvcc", machina::IsBuiltWithNvcc);
  m.def("IsAArch32Available", machina::IsAArch32Available);
  m.def("IsAArch64Available", machina::IsAArch64Available);
  m.def("IsPowerPCAvailable", machina::IsPowerPCAvailable);
  m.def("IsSystemZAvailable", machina::IsSystemZAvailable);
  m.def("IsX86Available", machina::IsX86Available);
  m.def("GpuSupportsHalfMatMulAndConv",
        machina::GpuSupportsHalfMatMulAndConv);
  m.def("IsMklEnabled", machina::IsMklEnabled);
}
