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

#include "machina/core/tpu/tpu_embedding_spmd_sharding_utils.h"

#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "machina/xla/hlo/builder/xla_builder.h"
#include "machina/xla/shape.h"
#include "machina/xla/tsl/platform/logging.h"  // IWYU pragma: keep
#include "machina/xla/xla_data.pb.h"
#include "machina/core/platform/statusor.h"

namespace machina {
namespace tpu {

absl::StatusOr<xla::OpSharding> SpmdShardingAnnotationOnFirstDim(
    const xla::Shape& shape, int core_count_per_replica,
    xla::XlaBuilder* builder) {
  if (!shape.IsArray()) {
    LOG(ERROR) << "Input shape is not ArrayType";
  }
  if (!shape.is_static()) {
    LOG(ERROR) << "Input shape is not static shape.";
  }

  xla::OpSharding op_sharding;
  if (shape.dimensions().empty()) {
    // Replicate scalar tensor (used for handling dynamic learning rates).
    op_sharding.set_type(xla::OpSharding::REPLICATED);
  } else {
    // Split tensors with rank >= 1 (used for embedding activations, gradients,
    // and deduplication data).
    if (shape.dimensions(0) % core_count_per_replica != 0) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Number of elements %d in the split dimension must be a multiple of "
          "the number of cores per replica %d",
          shape.dimensions(0), core_count_per_replica));
    }

    std::vector<int> tile_assignment_dimensions(shape.dimensions().size(), 1);
    tile_assignment_dimensions[0] = core_count_per_replica;

    op_sharding.set_type(xla::OpSharding::OTHER);
    for (const int tile_assignment : tile_assignment_dimensions) {
      op_sharding.add_tile_assignment_dimensions(tile_assignment);
    }
    for (int i = 0; i < core_count_per_replica; ++i) {
      op_sharding.add_tile_assignment_devices(i);
    }
  }
  return op_sharding;
}

}  // namespace tpu
}  // namespace machina
