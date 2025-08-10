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

#ifndef MACHINA_COMPILER_TF2MACHINA_MACHINA_XLA_KERNELS_TENSOR_LIST_UTILS_H_
#define MACHINA_COMPILER_TF2MACHINA_MACHINA_XLA_KERNELS_TENSOR_LIST_UTILS_H_

#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "machina/compiler/tf2xla/xla_op_kernel.h"
#include "machina/xla/hlo/builder/xla_builder.h"
#include "machina/xla/shape.h"
#include "machina/core/framework/tensor_shape.h"
#include "machina/core/platform/status.h"

namespace machina {

// Whether the input expression at `index` corresponds to a TensorList.
bool IsTensorListInput(XlaOpKernelContext* ctx, int index);

// Whether the TensorList is initialized (has known data type and shape).
absl::Status IsTensorListInitialized(xla::XlaOp list, bool* is_initialized);

// Whether the TensorList is a nested TensorList.
// Input must be an initialized TensorList.
// Non-nested and nested TensorLists are both supported.
absl::Status IsNestedTensorList(xla::XlaOp list, bool* is_nested_list);

// Builds a non-nested TensorList from `buffer` and `push_index`.
absl::Status BuildNonNestedTensorList(xla::XlaOp buffer, xla::XlaOp push_index,
                                      xla::XlaOp* output_list);

// Returns buffer shape for the TensorList.
// Input must be an initialized TensorList.
// Non-nested and nested TensorLists are both supported.
absl::Status GetTensorListBufferShape(xla::XlaOp list,
                                      xla::Shape* buffer_shape);

// Returns buffer for the TensorList.
// Input must be an initialized TensorList.
// Non-nested and nested TensorLists are both supported.
absl::Status GetTensorListBuffer(xla::XlaOp list, xla::XlaOp* buffer);

// Returns push index for the TensorList.
// Input must be an initialized TensorList.
// Non-nested and nested TensorLists are both supported.
absl::Status GetTensorListPushIndex(xla::XlaOp list, xla::XlaOp* push_index);

// Returns a new TensorList with given push_index.
// Input must be an initialized TensorList.
// Non-nested and nested TensorLists are both supported.
absl::Status SetTensorListPushIndex(xla::XlaOp list, xla::XlaOp push_index,
                                    xla::XlaOp* result);

// Returns an uninitialized TensorList.
xla::XlaOp BuildUninitializedTensorList(xla::XlaBuilder* b,
                                        int64_t leading_dimension,
                                        bool leading_size_is_dynamic,
                                        xla::XlaOp leading_dim_size);

// Returns leading dimension for the TensorList as well as a dynamic op
// representing the dynamic size. Input can be initialized or uninitialized
// TensorList. Non-nested and nested TensorLists are both supported.
absl::Status GetLeadingDimForTensorList(xla::XlaOp list, int64_t* leading_dim,
                                        bool* leading_dim_is_dynamic,
                                        xla::XlaOp* leading_dim_dynamic_size);

// Returns TensorList shape for the element shape.
// Element shape must be a normal tensor shape.
absl::Status GetTensorListShapeFromElementShape(const xla::Shape& element_shape,
                                                int64_t leading_dim,
                                                bool leading_dim_is_dynamic,
                                                xla::Shape* tensor_list_shape);

// Returns a TensorList filled by zeros with the given shape.
absl::Status CreateZerosTensorListWithShape(
    xla::XlaBuilder* b, const xla::Shape& list_shape,
    const std::vector<std::vector<xla::XlaOp>>& dynamic_dims, xla::XlaOp* list);

// If the TensorList is initialized, check that its shape matches element shape;
// If the TensorList is uninitialized, initialize it with the element shape.
// Input can be initialized or uninitialized TensorList.
// "element" can be normal tensor or TensorList.
absl::Status GetInitializedTensorListForElement(xla::XlaOp list,
                                                xla::XlaOp element,
                                                bool element_is_tensor_list,
                                                xla::XlaOp* initialized_list);

// Executes TensorListPushBack with given TensorList and element.
// Input must be an initialized TensorList.
// Non-nested and nested TensorLists are both supported.
absl::Status ExecuteTensorListPushBack(xla::XlaOp list, xla::XlaOp element,
                                       bool element_is_tensor_list,
                                       xla::XlaOp* result);

// Executes TensorListPopBack with given TensorList.
// Input must be an initialized TensorList.
// Non-nested and nested TensorLists are both supported.
absl::Status ExecuteTensorListPopBack(xla::XlaOp list, xla::XlaOp* list_result,
                                      xla::XlaOp* element_result,
                                      bool* element_is_tensor_list);

// Executes TensorListSetItem with given TensorList, index and element.
// Input must be an initialized TensorList.
// Only non-nested TensorList is supported.
absl::Status ExecuteTensorListSetItem(xla::XlaOp list, xla::XlaOp index,
                                      xla::XlaOp element, xla::XlaOp* result);

// Executes TensorListGetItem with given TensorList and index.
// Input must be an initialized TensorList.
// Only non-nested TensorList is supported.
absl::Status ExecuteTensorListGetItem(xla::XlaOp list, xla::XlaOp index,
                                      xla::XlaOp* result);

// Executes TensorListPushBack with given tensor and push index.
// "tensor" must be a normal tensor.
absl::Status ExecuteTensorListFromTensor(int push_index, xla::XlaOp tensor,
                                         xla::XlaOp* result);

}  // namespace machina

#endif  // MACHINA_COMPILER_TF2MACHINA_MACHINA_XLA_KERNELS_TENSOR_LIST_UTILS_H_
