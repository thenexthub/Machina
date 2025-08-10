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

// XLA-specific base classes for Reduction Ops.

#ifndef MACHINA_COMPILER_TF2MACHINA_MACHINA_XLA_KERNELS_REDUCTION_OPS_H_
#define MACHINA_COMPILER_TF2MACHINA_MACHINA_XLA_KERNELS_REDUCTION_OPS_H_

#include <cstdint>
#include <vector>

#include "machina/compiler/tf2xla/xla_op_kernel.h"
#include "machina/xla/hlo/builder/xla_builder.h"
#include "machina/xla/xla_data.pb.h"
#include "machina/core/framework/op_kernel.h"
#include "machina/core/framework/types.pb.h"

namespace machina {

// Reduction operations. The base class contains pure virtual methods
// to override: description is a textual description of the mapped
// function; InitialValue constructs the base case for the reduction;
// BuildReducer adds the implementation of the reduction lambda to a
// xla::XlaBuilder and BuildFinalizer adds the
// implementation of the finalizer lambda (if there is one) to a
// xla::XlaBuilder.
class XlaReductionOp : public XlaOpKernel {
 public:
  XlaReductionOp(OpKernelConstruction* ctx, DataType reduction_type);
  ~XlaReductionOp() override = default;

  // Return the base case for the reduction.
  virtual xla::XlaOp InitialValue(xla::XlaBuilder* builder) = 0;

  // Implement the (scalar,scalar)->scalar lambda that should be
  // applied to each pair of elements to be reduced. The desired
  // computation should be added to 'builder' and
  // '(scalar_lhs,scalar_rhs)' are the function's inputs.
  virtual void BuildReducer(xla::XlaBuilder* builder,
                            const xla::XlaOp& scalar_lhs,
                            const xla::XlaOp& scalar_rhs) = 0;

  // Applies a transformation to the output of the reduction. The desired
  // computation should be added to 'builder'. Argument 'input' is the original
  // input of the reduction; 'reduce_output' is the output of the reduction.
  // Returns the transformed reduction output. Defaults to returning
  // 'reduce_output' converted to the input type.
  virtual xla::XlaOp BuildFinalizer(
      xla::XlaBuilder* builder, const xla::XlaOp& input,
      const xla::XlaOp& reduce_output,
      const std::vector<int64_t>& dimensions_to_reduce);

  void Compile(XlaOpKernelContext* ctx) override;

 private:
  // True if the number of dimensions should be maintained.
  bool keep_dims_;

 protected:
  DataType reduction_type_;
  xla::PrimitiveType xla_reduction_type_;
};

}  // namespace machina

#endif  // MACHINA_COMPILER_TF2MACHINA_MACHINA_XLA_KERNELS_REDUCTION_OPS_H_
