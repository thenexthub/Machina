/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, July 12, 2025.
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
#include <string>
#include <vector>

#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/stl.h"  // from @pybind11
#include "machina/core/tpu/kernels/sparse_core_ops_utils.h"
#include "machina/python/lib/core/pybind11_lib.h"

PYBIND11_MODULE(_pywrap_tpu_embedding, m) {
  m.def("stack_tables",
        [](const std::vector<int64_t>& table_heights,
           const std::vector<int64_t>& table_widths,
           const std::vector<int64_t>& table_num_samples,
           const std::vector<int64_t>& table_groups,
           const std::vector<std::string>& table_names, int64_t num_tpu_chips) {
          return machina::GetTableStacks(table_heights, table_widths,
                                            table_num_samples, table_groups,
                                            table_names, num_tpu_chips);
        });
}
