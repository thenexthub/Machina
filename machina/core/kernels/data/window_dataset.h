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
#ifndef MACHINA_CORE_KERNELS_DATA_WINDOW_DATASET_H_
#define MACHINA_CORE_KERNELS_DATA_WINDOW_DATASET_H_

#include <vector>

#include "machina/core/framework/dataset.h"
#include "machina/core/framework/partial_tensor_shape.h"
#include "machina/core/framework/tensor.h"
#include "machina/core/framework/tensor_shape.h"
#include "machina/core/framework/types.h"
#include "machina/core/platform/status.h"

namespace machina {
namespace data {

// Creates a dataset representing an eagerly-collected window of elements.
//
// The `elements` argument defines the elements of the resulting
// dataset, which is stored in `out_dataset`.
//
// This dataset is constructed internally for use in datasets that
// build nested dataset expressions (e.g. the reducer function for
// GroupByWindowDataset). It efficiently supports multiple iterators on
// the same window without recomputation.
//
// REQUIRES: `output_types` must match the types of the respective
// element components in `elements`.
// REQUIRES: `output_shapes` must be compatible with the shapes of the
// respective element components in `elements`.a
absl::Status NewWindow(std::vector<std::vector<Tensor>> elements,
                       DataTypeVector output_types,
                       std::vector<PartialTensorShape> output_shapes,
                       DatasetBase** out_dataset);

}  // namespace data
}  // namespace machina

#endif  // MACHINA_CORE_KERNELS_DATA_WINDOW_DATASET_H_
