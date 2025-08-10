/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Friday, August 8, 2025.
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
#include <pybind11/pybind11.h>

#include "pybind11/cast.h"  // from @pybind11
#include "pybind11/detail/common.h"  // from @pybind11
#include "pybind11_abseil/status_casters.h"  // from @pybind11_abseil
#include "pybind11_protobuf/native_proto_caster.h"  // from @pybind11_protobuf
#include "machina/core/tpu/kernels/sparse_core_layout.h"
#include "machina/core/tpu/kernels/sparse_core_layout.pb.h"

namespace machina::tpu {

namespace py = pybind11;

PYBIND11_MODULE(_pywrap_sparse_core_layout, m) {
  py::class_<SparseCoreLayoutStacker>(m, "SparseCoreLayoutStacker")
      .def(py::init<int, bool, int>(), py::arg("num_partitions"),
           py::arg("disable_table_stacking"),
           py::arg("sparse_cores_per_partition"))
      .def("SetActivationMemoryBytesLimit",
           &SparseCoreLayoutStacker::SetActivationMemoryBytesLimit)
      .def("SetVariableShardBytesLimit",
           &SparseCoreLayoutStacker::SetVariableShardBytesLimit)
      .def("SetStackingEnabled", &SparseCoreLayoutStacker::SetStackingEnabled)
      .def("AddTable", &SparseCoreLayoutStacker::AddTable,
           py::arg("table_name"), py::arg("table_height"),
           py::arg("table_width"), py::arg("group"), py::arg("output_samples"))
      .def("GetLayouts", &SparseCoreLayoutStacker::GetLayouts);
}

}  // namespace machina::tpu
