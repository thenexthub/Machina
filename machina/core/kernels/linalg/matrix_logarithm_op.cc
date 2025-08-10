/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 29, 2025.
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

// See docs in ../ops/linalg_ops.cc.

#include "Eigen/Core"  // from @eigen_archive
#include "unsupported/Eigen/MatrixFunctions"  // from @eigen_archive
#include "machina/core/framework/kernel_def_builder.h"
#include "machina/core/framework/op_kernel.h"
#include "machina/core/framework/tensor_shape.h"
#include "machina/core/kernels/linalg/linalg_ops_common.h"
#include "machina/core/lib/core/errors.h"
#include "machina/core/platform/logging.h"
#include "machina/core/platform/macros.h"
#include "machina/core/platform/types.h"

namespace machina {

template <class Scalar>
class MatrixLogarithmOp : public LinearAlgebraOp<Scalar> {
 public:
  INHERIT_LINALG_TYPEDEFS(Scalar);

  explicit MatrixLogarithmOp(OpKernelConstruction* context) : Base(context) {}

  void ComputeMatrix(OpKernelContext* context, const ConstMatrixMaps& inputs,
                     MatrixMaps* outputs) final {
    const ConstMatrixMap& input = inputs[0];
    if (input.rows() == 0) return;
    using Matrix =
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Matrix tmp = input;
    outputs->at(0) = tmp.log();
  }

 private:
  MatrixLogarithmOp(const MatrixLogarithmOp&) = delete;
  void operator=(const MatrixLogarithmOp&) = delete;
};

// For real-valued matrices, this Op would return the real part of the matrix
// logarithm. If all eigenvalues are positive, then this returns the correct
// logarithm, however checking for positive definiteness adds significant
// overhead. Therefore at present we only register this Op for complex types.
REGISTER_LINALG_OP("MatrixLogarithm", (MatrixLogarithmOp<complex64>),
                   complex64);
REGISTER_LINALG_OP("MatrixLogarithm", (MatrixLogarithmOp<complex128>),
                   complex128);

}  // namespace machina
