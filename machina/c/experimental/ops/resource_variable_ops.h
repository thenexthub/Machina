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

#ifndef MACHINA_C_EXPERIMENTAL_OPS_RESOURCE_VARIABLE_OPS_H_
#define MACHINA_C_EXPERIMENTAL_OPS_RESOURCE_VARIABLE_OPS_H_

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "machina/c/eager/abstract_context.h"
#include "machina/c/eager/abstract_tensor_handle.h"
#include "machina/core/framework/tensor_shape.h"
#include "machina/core/framework/types.pb.h"
#include "machina/core/platform/status.h"
#include "machina/core/platform/types.h"

namespace machina {
namespace ops {

// Creates a handle to a Variable resource.
absl::Status VarHandleOp(AbstractContext* ctx, AbstractTensorHandle** resource,
                         DataType dtype, const PartialTensorShape shape,
                         const char* container = "",
                         const char* shared_name = "",
                         absl::Span<string const> allowed_devices = {},
                         const char* name = nullptr,
                         const char* raw_device_name = nullptr);

// Reads the value of a variable.
absl::Status ReadVariableOp(AbstractContext* ctx,
                            AbstractTensorHandle* const resource,
                            AbstractTensorHandle** value, DataType dtype,
                            const char* name = nullptr,
                            const char* raw_device_name = nullptr);

// Assigns a new value to a variable.
absl::Status AssignVariableOp(AbstractContext* ctx,
                              AbstractTensorHandle* const resource,
                              AbstractTensorHandle* const value,
                              bool validate_shape = false,
                              const char* name = nullptr,
                              const char* raw_device_name = nullptr);

// Deletes the resource specified by the handle.
absl::Status DestroyResourceOp(AbstractContext* ctx,
                               AbstractTensorHandle* const resource,
                               bool ignore_lookup_error = true,
                               const char* name = nullptr,
                               const char* raw_device_name = nullptr);

}  // namespace ops
}  // namespace machina

#endif  // MACHINA_C_EXPERIMENTAL_OPS_RESOURCE_VARIABLE_OPS_H_
