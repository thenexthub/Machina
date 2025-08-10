/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, March 6, 2025.
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

#ifndef MACHINA_CORE_KERNELS_SAVE_RESTORE_TENSOR_H_
#define MACHINA_CORE_KERNELS_SAVE_RESTORE_TENSOR_H_

#include "machina/core/util/tensor_slice_reader.h"
#include "machina/core/util/tensor_slice_writer.h"

namespace machina {

class OpKernelContext;

// Legacy / V1 checkpoint format.

// Save input tensors in *context to a writer built from builder_func().
// context must have the following inputs:
//  0: a single element string tensor that contains the file name.
//  1: names for the remaining tensors
// If save_slices is true:
//  2: shape and slice specifications.
//  rest: tensors to save
void SaveTensors(
    OpKernelContext* context,
    checkpoint::TensorSliceWriter::CreateBuilderFunction builder_func,
    bool save_slices);

// Reads a single tensor from the reader built from open_func() and produces
// it as context->output(restore_index).  "preferred_shard" is the same the
// TensorSliceReader preferred_shard parameter.
//
// context must have the following inputs:
//  0: a single element string tensor that contains the file name.
//  1: string tensor that names the outputs to be restored.
// If restore_slice is true:
//  2: shape and slice specification of the tensors to restore.
//
// restore_index indicates the variable name and slice to lookup
// in context(1) and (2).
void RestoreTensor(OpKernelContext* context,
                   checkpoint::TensorSliceReader::OpenTableFunction open_func,
                   int preferred_shard, bool restore_slice, int restore_index);

// V2 checkpoint format.

// Invokes the V2 checkpoint read path to read tensors.
//
// "context" is only used for allocating outputs.  In particular, the inputs are
// explicitly provided and not accessed via the "input(i)" methods.
// REQUIRES:
//   * "prefix" has 1 element, DT_STRING.
//   * "tensor_names" and "shape_and_slices" shaped {N}, both DT_STRING.
//   * "dtypes" has N elements, the datatypes of the to-restore tensors.
absl::Status RestoreTensorsV2(OpKernelContext* context, const Tensor& prefix,
                              const Tensor& tensor_names,
                              const Tensor& shape_and_slices,
                              absl::Span<const DataType> dtypes);

}  // namespace machina

#endif  // MACHINA_CORE_KERNELS_SAVE_RESTORE_TENSOR_H_
