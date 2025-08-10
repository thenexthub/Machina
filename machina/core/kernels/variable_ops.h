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

#ifndef MACHINA_CORE_KERNELS_VARIABLE_OPS_H_
#define MACHINA_CORE_KERNELS_VARIABLE_OPS_H_

#include "machina/core/framework/allocator.h"
#include "machina/core/framework/op_kernel.h"
#include "machina/core/framework/register_types.h"
#include "machina/core/framework/resource_mgr.h"
#include "machina/core/framework/resource_var.h"
#include "machina/core/lib/core/errors.h"
#include "machina/core/platform/macros.h"
#include "machina/core/platform/mutex.h"
#include "machina/core/platform/types.h"

namespace machina {

class VariableOp : public OpKernel {
 public:
  explicit VariableOp(OpKernelConstruction* context);
  void Compute(OpKernelContext* ctx) override;

 private:
  DataType dtype_;
  TensorShape shape_;
  ContainerInfo cinfo_;

  VariableOp(const VariableOp&) = delete;
  void operator=(const VariableOp&) = delete;
};

}  // namespace machina

#endif  // MACHINA_CORE_KERNELS_VARIABLE_OPS_H_
