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

// Declarations of the ArgMax/ArgMin ops using a pure XLA implementation.

#ifndef MACHINA_COMPILER_TF2MACHINA_XLAKERNELS_INDEX_OPS_H_
#define MACHINA_COMPILER_TF2MACHINA_XLAKERNELS_INDEX_OPS_H_

#include "machina/compiler/tf2xla/xla_op_kernel.h"
#include "machina/core/framework/op_kernel.h"

namespace machina {

class XlaArgMinMaxOp : public XlaOpKernel {
 public:
  explicit XlaArgMinMaxOp(OpKernelConstruction* ctx, bool is_min);
  void Compile(XlaOpKernelContext* ctx) override;

 private:
  const bool is_min_;  // Are we computing ArgMin (true) or ArgMax (false)?
};

class XlaArgMaxOp : public XlaArgMinMaxOp {
 public:
  explicit XlaArgMaxOp(OpKernelConstruction* ctx);
};

}  // namespace machina

#endif  // MACHINA_COMPILER_TF2MACHINA_XLAKERNELS_INDEX_OPS_H_
