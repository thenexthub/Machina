/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Saturday, June 21, 2025.
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

// This file is MACHINE GENERATED! Do not edit.

#ifndef MACHINA_C_EXPERIMENTAL_OPS_ARRAY_OPS_H_
#define MACHINA_C_EXPERIMENTAL_OPS_ARRAY_OPS_H_

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "machina/c/eager/abstract_context.h"
#include "machina/c/eager/abstract_tensor_handle.h"
#include "machina/core/framework/types.pb.h"
#include "machina/core/platform/status.h"

namespace machina {
namespace ops {

// Return a tensor with the same shape and contents as the input tensor or
// value.
absl::Status Identity(AbstractContext* ctx, AbstractTensorHandle* const input,
                      AbstractTensorHandle** output, const char* name = nullptr,
                      const char* raw_device_name = nullptr);

// Returns a list of tensors with the same shapes and contents as the input
absl::Status IdentityN(AbstractContext* ctx,
                       absl::Span<AbstractTensorHandle* const> input,
                       absl::Span<AbstractTensorHandle*> output,
                       const char* name = nullptr,
                       const char* raw_device_name = nullptr);

// Returns a tensor of zeros with the same shape and type as x.
absl::Status ZerosLike(AbstractContext* ctx, AbstractTensorHandle* const x,
                       AbstractTensorHandle** y, const char* name = nullptr,
                       const char* raw_device_name = nullptr);

// Returns the shape of a tensor.
absl::Status Shape(AbstractContext* ctx, AbstractTensorHandle* const input,
                   AbstractTensorHandle** output, DataType out_type = DT_INT32,
                   const char* name = nullptr,
                   const char* raw_device_name = nullptr);

// Inserts a dimension of 1 into a tensor's shape.
absl::Status ExpandDims(AbstractContext* ctx, AbstractTensorHandle* const input,
                        AbstractTensorHandle* const dim,
                        AbstractTensorHandle** output,
                        const char* name = nullptr,
                        const char* raw_device_name = nullptr);

// Returns a tensor of ones with the same shape and type as x.
absl::Status OnesLike(AbstractContext* ctx, AbstractTensorHandle* const x,
                      AbstractTensorHandle** y, const char* name = nullptr,
                      const char* raw_device_name = nullptr);

}  // namespace ops
}  // namespace machina

#endif  // MACHINA_C_EXPERIMENTAL_OPS_ARRAY_OPS_H_
