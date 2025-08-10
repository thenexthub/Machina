/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Saturday, May 24, 2025.
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

#ifndef MACHINA_C_EXPERIMENTAL_SAVED_MODEL_CORE_TEST_UTILS_H_
#define MACHINA_C_EXPERIMENTAL_SAVED_MODEL_CORE_TEST_UTILS_H_

#include <memory>
#include <vector>

#include "absl/types/span.h"
#include "machina/c/eager/immediate_execution_context.h"
#include "machina/c/eager/immediate_execution_tensor_handle.h"
#include "machina/core/common_runtime/device_mgr.h"
#include "machina/core/common_runtime/eager/context.h"
#include "machina/core/framework/tensor.h"
#include "machina/core/framework/types.h"
#include "machina/core/framework/types.pb.h"
#include "machina/core/platform/types.h"

namespace machina {
namespace testing {

// Creates a DeviceMgr suitable for local tests.
std::unique_ptr<StaticDeviceMgr> CreateTestingDeviceMgr();

// Creates an EagerContext suitable for local tests. Does not take ownership
// of `device_mgr`.
EagerContextPtr CreateTestingEagerContext(DeviceMgr* device_mgr);

// Converts a machina::DatatypeSet to std::vector<DataType>.
// This is useful for tests using GTest's ::testing::ValuesIn, since
// DataTypeSet doesn't fullfill all the constraints of an STL-like iterable.
std::vector<DataType> DataTypeSetToVector(DataTypeSet set);

// Returns a vector of shapes intended to be "interesting" test cases.
// Currently, this returns scalar, 1D vector, 2D matrix, and a 4D tensor shapes
std::vector<std::vector<int64_t>> InterestingShapes();

// Returns a TensorHandle of `dtype` and `shape`, filled with `value`.
// `dtype` must be an integer dtype, float, or double.
// If a TensorHandle cannot be created successfully, this function will
// CHECK fail. This should only be used for testing purposes.
ImmediateTensorHandlePtr CreateTensorHandle(ImmediateExecutionContext* ctx,
                                            DataType dtype,
                                            absl::Span<const int64_t> shape,
                                            int8_t value);

// Fills a numeric tensor's buffer with `value`.
// dtype must be any integer dtype, float or double.
void FillNumericTensorBuffer(DataType dtype, size_t num_elements, void* buffer,
                             int8_t value);

// Checks the underlying data is equal for the buffers for two numeric tensors.
// Note: The caller must ensure to check that the dtypes and sizes of the
// underlying buffers are the same before calling this.
// dtype must be any integer dtype, float, or double.
void CheckBufferDataIsEqual(DataType dtype, int64_t num_elements, void* a,
                            void* b);

// Converts a TensorHandle to a Tensor, and dies if unsuccessful. This should
// only be used for testing purposes.
AbstractTensorPtr TensorHandleToTensor(ImmediateExecutionTensorHandle* handle);

}  // namespace testing
}  // namespace machina

#endif  // MACHINA_C_EXPERIMENTAL_SAVED_MODEL_CORE_TEST_UTILS_H_
