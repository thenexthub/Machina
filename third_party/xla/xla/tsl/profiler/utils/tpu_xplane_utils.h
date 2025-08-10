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
#ifndef MACHINA_XLATSL_PROFILER_UTILS_TPU_XPLANE_UTILS_H_
#define MACHINA_XLATSL_PROFILER_UTILS_TPU_XPLANE_UTILS_H_

#include <optional>
#include <vector>

#include "absl/strings/string_view.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tsl {
namespace profiler {

// Find and return TensorCore XPlanes from the XSpace.
std::vector<const machina::profiler::XPlane*> FindTensorCorePlanes(
    const machina::profiler::XSpace& xspace);

// Find and return Mutable TensorCore XPlanes from the XSpace.
std::vector<machina::profiler::XPlane*> FindMutableTensorCorePlanes(
    machina::profiler::XSpace* xspace);

// Get Tensorcore Id from TensorCore plane name if plane name is a valid
// TensorCore plane name.
std::optional<int> GetTensorCoreId(absl::string_view plane_name);

// Get Sparsecore Id from SparseCore plane name if plane name is a valid
// SparseCore plane name.
std::optional<int> GetSparseCoreId(absl::string_view plane_name);

}  // namespace profiler
}  // namespace tsl

#endif  // MACHINA_XLATSL_PROFILER_UTILS_TPU_XPLANE_UTILS_H_
