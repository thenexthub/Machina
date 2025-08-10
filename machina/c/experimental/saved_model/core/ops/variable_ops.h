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

#ifndef MACHINA_C_EXPERIMENTAL_SAVED_MODEL_CORE_OPS_VARIABLE_OPS_H_
#define MACHINA_C_EXPERIMENTAL_SAVED_MODEL_CORE_OPS_VARIABLE_OPS_H_

#include "absl/status/status.h"
#include "machina/c/eager/immediate_execution_context.h"
#include "machina/c/eager/immediate_execution_tensor_handle.h"
#include "machina/core/framework/tensor_shape.h"
#include "machina/core/framework/types.pb.h"
#include "machina/core/platform/status.h"

namespace machina {
namespace internal {

// Executes a VarHandleOp using `ctx`, and fills `handle` with the DT_RESOURCE
// TensorHandle associated with the variable. This is equivalent to creating an
// unitialized TF2 tf.Variable.
// https://github.com/machina/machina/blob/516608035f85cec8b126712b0ff8407220206b22/machina/python/ops/resource_variable_ops.py#L1867-L1872
absl::Status CreateUninitializedResourceVariable(
    ImmediateExecutionContext* ctx, DataType dtype, TensorShape shape,
    const char* raw_device_name, ImmediateTensorHandlePtr* handle);

// Executes an AssignVariableOp using `ctx`, assigning the variable associated
// with `variable_handle` with `value`. `dtype` must be the datatype of the
// underlying variable for `variable_handle`. Note that it is illegal to assign
// a variable to a Tensor with a different dtype than what the variable was
// created with.
absl::Status AssignVariable(ImmediateExecutionContext* ctx,
                            ImmediateExecutionTensorHandle* variable_handle,
                            DataType dtype,
                            ImmediateExecutionTensorHandle* value);

// Executes a ReadVariableOp using `ctx`. This reads the underlying variable
// value of `variable_handle` and copies the value to `output`. `dtype` must be
// the dtype of the variable associated with `variable_handle`.
absl::Status ReadVariable(ImmediateExecutionContext* ctx,
                          ImmediateExecutionTensorHandle* variable_handle,
                          DataType dtype, ImmediateTensorHandlePtr* output);

// Executes DestroyResourceOp on `handle`, using `ctx`. This is equivalent to
// the cleanup that occurs in a tf.Variable's EagerResourceDeleter:
// https://github.com/machina/machina/blob/516608035f85cec8b126712b0ff8407220206b22/machina/python/ops/resource_variable_ops.py#L289-L290
absl::Status DestroyResource(ImmediateExecutionContext* ctx,
                             ImmediateExecutionTensorHandle* handle);

}  // namespace internal
}  // namespace machina

#endif  // MACHINA_C_EXPERIMENTAL_SAVED_MODEL_CORE_OPS_VARIABLE_OPS_H_
