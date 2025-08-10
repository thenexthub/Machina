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

// See docs in ../ops/linalg_ops.cc.

#include "Eigen/Cholesky"  // from @eigen_archive
#include "Eigen/Core"  // from @eigen_archive
#include "machina/core/framework/kernel_def_builder.h"
#include "machina/core/framework/op_kernel.h"
#include "machina/core/framework/register_types.h"
#include "machina/core/framework/tensor_shape.h"
#include "machina/core/kernels/linalg/linalg_ops_common.h"
#include "machina/core/lib/core/errors.h"
#include "machina/core/platform/logging.h"
#include "machina/core/platform/types.h"

namespace machina {

template <class Scalar>
class CholeskyOp : public LinearAlgebraOp<Scalar> {
 public:
  INHERIT_LINALG_TYPEDEFS(Scalar);

  explicit CholeskyOp(OpKernelConstruction* context) : Base(context) {}

  void ComputeMatrix(OpKernelContext* context, const ConstMatrixMaps& inputs,
                     MatrixMaps* outputs) final {
    const ConstMatrixMap& input = inputs[0];
    if (input.rows() == 0) {
      // If X is an empty matrix (0 rows, 0 col), X * X' == X.
      // Therefore, we return X.
      return;
    }
    // Perform the actual LL^T Cholesky decomposition. This will only use
    // the lower triangular part of data_in by default. The upper triangular
    // part of the matrix will not be read.
    Eigen::LLT<
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        llt_decomposition(input);

    // If decomposition fails, fill output with NaNs so the failure can
    // be detected at runtime.
    if (!TF_PREDICT_TRUE(llt_decomposition.info() == Eigen::Success)) {
      LOG(WARNING) << "Cholesky decomposition was not successful. "
                      "Eigen::LLT failed with error code "
                   << llt_decomposition.info()
                   << ". Filling lower-triangular output with NaNs.";
      outputs->at(0).template triangularView<Eigen::Lower>().fill(
          Eigen::NumTraits<Scalar>::quiet_NaN());
    } else {
      // Output the lower triangular in a dense form.
      outputs->at(0) = llt_decomposition.matrixL();
    }
  }
};

REGISTER_LINALG_OP("Cholesky", (CholeskyOp<float>), float);
REGISTER_LINALG_OP("Cholesky", (CholeskyOp<double>), double);
REGISTER_LINALG_OP("Cholesky", (CholeskyOp<complex64>), complex64);
REGISTER_LINALG_OP("Cholesky", (CholeskyOp<complex128>), complex128);
REGISTER_LINALG_OP("BatchCholesky", (CholeskyOp<float>), float);
REGISTER_LINALG_OP("BatchCholesky", (CholeskyOp<double>), double);

}  // namespace machina
