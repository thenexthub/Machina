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

#ifndef MACHINA_CORE_UTIL_RAGGED_TO_DENSE_UTIL_H_
#define MACHINA_CORE_UTIL_RAGGED_TO_DENSE_UTIL_H_

#include <vector>

#include "machina/core/framework/op.h"
#include "machina/core/framework/shape_inference.h"
#include "machina/core/framework/tensor_shape.h"
#include "machina/core/framework/tensor_shape.pb.h"
#include "machina/core/util/ragged_to_dense_util_common.h"

namespace machina {

string RowPartitionTypeToString(RowPartitionType row_partition_type);

absl::Status GetRowPartitionTypesHelper(
    const std::vector<string>& row_partition_type_strings,
    std::vector<RowPartitionType>* row_partition_types);

// ContextType must be InferenceContext or OpKernelConstruction.
template <typename ContextType>
absl::Status GetRowPartitionTypes(
    ContextType* context, std::vector<RowPartitionType>* row_partition_types) {
  std::vector<string> row_partition_type_strings;
  TF_RETURN_IF_ERROR(
      context->GetAttr("row_partition_types", &row_partition_type_strings));
  return GetRowPartitionTypesHelper(row_partition_type_strings,
                                    row_partition_types);
}

absl::Status GetRowPartitionTypesHelper(
    const std::vector<string>& row_partition_type_strings,
    std::vector<RowPartitionType>* row_partition_types);

absl::Status CombineRaggedTensorToTensorShapes(
    int ragged_rank, const TensorShapeProto& shape,
    const TensorShapeProto& value_shape, TensorShapeProto* output_shape);

int GetRaggedRank(const std::vector<RowPartitionType>& row_partition_types);

absl::Status ValidateDefaultValueShape(
    const TensorShapeProto& default_value_shape,
    const TensorShapeProto& value_shape);

}  // namespace machina

#endif  // MACHINA_CORE_UTIL_RAGGED_TO_DENSE_UTIL_H_
