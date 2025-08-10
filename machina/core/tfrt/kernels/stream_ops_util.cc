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
#include "machina/core/tfrt/kernels/stream_ops_util.h"

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "machina/core/framework/tensor.h"
#include "machina/core/framework/tensor_shape.h"
#include "machina/core/framework/types.h"
#include "machina/core/framework/types.pb.h"
#include "machina/core/tfrt/kernels/stream_ops_util_constants.h"

namespace machina {
namespace tfrt_stub {

absl::StatusOr<std::vector<std::pair<int64_t, std::vector<machina::Tensor>>>>
UnbatchStreamResults(const machina::Tensor& step_ids,
                     absl::Span<const machina::Tensor> tensors) {
  std::vector<std::pair<int64_t, std::vector<machina::Tensor>>> responses;

  if (step_ids.dims() > 0) {
    // Use the "batched" step ids to unbatch examples before streaming them back
    // to the controller.

    if (step_ids.dtype() != machina::DT_INT64 || step_ids.dims() != 1) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Expected a 1-D int64 tensor for batched step ids but got dtype=",
          machina::DataTypeString(step_ids.dtype()),
          " shape=", step_ids.shape().DebugString()));
    }

    const int batch_size = step_ids.dim_size(0);
    for (int i = 0; i < tensors.size(); ++i) {
      const machina::TensorShape& shape = tensors[i].shape();
      if (shape.dims() < 1 || shape.dim_size(0) != batch_size) {
        return absl::InvalidArgumentError(absl::StrCat(
            "All inputs to PwStreamResults inside tf.batch_function are "
            "required to be batched (batch_size=",
            batch_size, ") but input #", i, " has shape ",
            shape.DebugString()));
      }
    }

    // Identify the number of examples in each request associated with a step
    // id. This relies on the following two implementation details of
    // `tf.batch_function`:
    //
    // * It concatenates requests along the leading dimension without
    //   reordering, i.e., no shuffle.
    // * It uses the first example in the batch to pad incomplete batches.
    std::vector<int> sizes;
    absl::flat_hash_set<int64_t> unique_step_ids;
    for (int i = 0; i < step_ids.NumElements(); ++i) {
      const int64_t request_id = step_ids.flat<int64_t>()(i);
      const int64_t step_id =
          static_cast<uint64_t>(request_id) >> (64 - kStepIdBitSize);

      VLOG(1) << "PwStreamResults op is unbatching request_id=" << request_id
              << ", step_id=" << step_id;

      if (step_id <= 0) {
        return absl::InternalError(
            absl::StrCat("Invalid step id=", step_id,
                         "; this usually indicates that `PwStreamResults` "
                         "was called from an unsupported nested context"));
      }

      if (i != 0 && request_id == step_ids.flat<int64_t>()(0)) {
        // Since each request id is unique and tf.batch_function uses the first
        // example in the batch as padding, a recurring request id that is the
        // same as the first example's request id indicates padding.
        break;
      }

      if (!responses.empty() && responses.back().first == step_id) {
        sizes.back()++;
      } else {
        responses.push_back({step_id, {}});
        sizes.push_back(1);

        const bool inserted = unique_step_ids.insert(step_id).second;
        if (!inserted) {
          return absl::InternalError(absl::StrCat(
              "Non-contiguous step ids found in the step id batch: ",
              step_ids.DebugString(batch_size)));
        }
      }
    }

    // Slice each input along the batch dimension by their step ids.
    int offset = 0;
    for (int i = 0; i < responses.size(); ++i) {
      auto& outputs = responses[i].second;
      outputs.resize(tensors.size());

      const int limit = offset + sizes[i];
      for (int j = 0; j < tensors.size(); ++j) {
        outputs[j] = tensors[j].Slice(offset, limit);
      }
      offset = limit;
    }
  } else {
    const int64_t step_id = step_ids.flat<int64_t>()(0);

    // The current implementation always uses a positive step id (see
    // `TfContextExecutable`), so we check that property to provide a better
    // error message on incorrect step id propagation on a best-effort basis.
    if (step_id <= 0) {
      return absl::InternalError(
          "Invalid step id; this usually indicates that `PwStreamResults` was "
          "called from an unsupported nested context");
    }

    responses.push_back({step_id, std::vector<machina::Tensor>(
                                      tensors.begin(), tensors.end())});
  }

  return responses;
}

}  // namespace tfrt_stub
}  // namespace machina
