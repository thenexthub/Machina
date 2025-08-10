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
#ifndef MACHINA_C_EAGER_UNIFIED_API_TESTUTIL_H_
#define MACHINA_C_EAGER_UNIFIED_API_TESTUTIL_H_

#include "machina/c/eager/abstract_context.h"
#include "machina/c/eager/abstract_tensor_handle.h"
#include "machina/c/eager/c_api_test_util.h"
#include "machina/c/eager/c_api_unified_experimental.h"
#include "machina/c/eager/c_api_unified_experimental_internal.h"
#include "machina/c/tf_status_helper.h"
#include "machina/c/tf_tensor.h"
#include "machina/core/platform/status.h"

namespace machina {

// Builds and returns a `TracingContext` using the default tracing impl.
AbstractContext* BuildFunction(const char* fn_name);

// Creates parameters (placeholders) in the tracing `ctx` using the shape and
// dtype of `inputs`.
absl::Status CreateParamsForInputs(
    AbstractContext* ctx, absl::Span<AbstractTensorHandle* const> inputs,
    std::vector<AbstractTensorHandle*>* params);

// A callable that takes tensor inputs and returns zero or more tensor outputs.
using Model = std::function<absl::Status(
    AbstractContext*, absl::Span<AbstractTensorHandle* const>,
    absl::Span<AbstractTensorHandle*>)>;

// Runs `model` maybe wrapped in a function call op. This can be thought as
// being equivalent to the following python code.
//
// if use_function:
//   outputs = tf.function(model)(inputs)
// else:
//   outputs = model(inputs)
absl::Status RunModel(Model model, AbstractContext* ctx,
                      absl::Span<AbstractTensorHandle* const> inputs,
                      absl::Span<AbstractTensorHandle*> outputs,
                      bool use_function);

absl::Status BuildImmediateExecutionContext(bool use_tfrt,
                                            AbstractContext** ctx);

// Return a tensor handle with given type, values and dimensions.
template <class T, TF_DataType datatype>
absl::Status TestTensorHandleWithDims(AbstractContext* ctx, const T* data,
                                      const int64_t* dims, int num_dims,
                                      AbstractTensorHandle** tensor) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  TFE_Context* eager_ctx =
      TF_ExecutionContextGetTFEContext(wrap(ctx), status.get());
  TF_RETURN_IF_ERROR(StatusFromTF_Status(status.get()));
  TFE_TensorHandle* input_eager =
      TestTensorHandleWithDims<T, datatype>(eager_ctx, data, dims, num_dims);
  *tensor =
      unwrap(TF_CreateAbstractTensorFromEagerTensor(input_eager, status.get()));
  return absl::OkStatus();
}

// Return a scalar tensor handle with given value.
template <class T, TF_DataType datatype>
absl::Status TestScalarTensorHandle(AbstractContext* ctx, const T value,
                                    AbstractTensorHandle** tensor) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  TFE_Context* eager_ctx =
      TF_ExecutionContextGetTFEContext(wrap(ctx), status.get());
  TF_RETURN_IF_ERROR(StatusFromTF_Status(status.get()));
  TFE_TensorHandle* input_eager =
      TestScalarTensorHandle<T, datatype>(eager_ctx, value);
  *tensor =
      unwrap(TF_CreateAbstractTensorFromEagerTensor(input_eager, status.get()));
  return absl::OkStatus();
}

// Places data from `t` into *result_tensor.
absl::Status GetValue(AbstractTensorHandle* t, TF_Tensor** result_tensor);
}  // namespace machina

#endif  // MACHINA_C_EAGER_UNIFIED_API_TESTUTIL_H_
