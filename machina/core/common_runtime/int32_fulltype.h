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

#ifndef MACHINA_CORE_COMMON_RUNTIME_INT32_FULLTYPE_H_
#define MACHINA_CORE_COMMON_RUNTIME_INT32_FULLTYPE_H_

#include "machina/core/graph/graph.h"
#include "machina/core/lib/core/status.h"
#include "machina/core/platform/macros.h"

namespace machina {

// An optimization (graph rewrite) pass to automatically set TFT_SHAPE_TENSOR
// full type information annotations for all int32 tensors, creating or
// modifying existing full type information as needed. This allows placement
// mechanisms using full type information to always place int32 on host.
class Int32FulltypePass {
 public:
  Int32FulltypePass() = default;
  explicit Int32FulltypePass(string debug_location)
      : debug_location_(debug_location) {}

  // For each node in this graph that outputs int32 tensors, set full
  // type information such that the int32 tensors use TFT_SHAPE_TENSOR
  // (or TFT_TENSOR if ints_on_device is true, which is only for single
  // device functions including the functions with just one op used for
  // eager execution).
  //
  // This method is not thread-safe.
  absl::Status ProcessGraph(Graph* graph, bool ints_on_device);

  // Update full type information for int32 tensors that are in HOST_MEMORY
  // to use TFT_SHAPE_TENSOR. The type_id of TENSOR_T is expected to be
  // TFT_UNSET, TFT_TENSOR or TFT_SHAPE_TENSOR on input and will be updated
  // to TFT_SHAPE_TENSOR on output for int32 tensors if it is not
  // TFT_SHAPE_TENSOR already. For tensors that are not int32, if the input full
  // type information is TFT_UNSET, it will only be updated if SET_ONLY_INT32 is
  // false. Note that TENSOR_T is not the full type information for the outputs
  // of a node, so it does have an outer TFT_PRODUCT. NODE and OUTPUT_IDX are
  // optional and only used in an error message to say that the tensor is output
  // OUTPUT_IDX of node NODE.
  absl::Status Int32FullTypeForTensor(DataType dtype, FullTypeDef* tensor_t,
                                      bool set_only_int32, Node* node = nullptr,
                                      int output_idx = 0);

 private:
  // Location of where annotations were added for debug messages.
  string debug_location_;
};

}  // namespace machina

#endif  // MACHINA_CORE_COMMON_RUNTIME_INT32_FULLTYPE_H_
