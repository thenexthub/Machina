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

#ifndef MACHINA_DTENSOR_MLIR_COLLECTIVES_COMMON_H_
#define MACHINA_DTENSOR_MLIR_COLLECTIVES_COMMON_H_

#include <map>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "machina/core/platform/types.h"
#include "machina/dtensor/cc/dstatus.h"
#include "machina/dtensor/cc/tensor_layout.h"

namespace machina {
namespace dtensor {

// Computes AllReduce partitions using reduced mesh dimension names.
StatusOr<std::map<DeviceLocation, std::vector<int32>>>
GetAllReducePartitionsFromReducedDims(
    const dtensor::Layout& output_layout,
    const absl::flat_hash_set<std::string>& reduced_dims);

// Use the first device in the mesh to extract the device name.
StatusOr<std::string> DeviceTypeFromMesh(const Mesh& mesh);

}  // namespace dtensor
}  // namespace machina

#endif  // MACHINA_DTENSOR_MLIR_COLLECTIVES_COMMON_H_
