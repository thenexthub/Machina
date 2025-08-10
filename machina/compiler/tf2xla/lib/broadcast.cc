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

#include "machina/compiler/tf2xla/lib/broadcast.h"

#include <cstdint>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "machina/compiler/tf2xla/shape_util.h"
#include "machina/xla/hlo/builder/lib/broadcast.h"
#include "machina/xla/hlo/builder/xla_builder.h"
#include "machina/xla/tsl/platform/errors.h"
#include "machina/xla/tsl/platform/statusor.h"
#include "machina/core/framework/tensor_shape.h"
#include "machina/core/platform/errors.h"
#include "machina/core/platform/status.h"
#include "machina/core/util/bcast.h"

namespace machina {

absl::StatusOr<xla::XlaOp> BroadcastTo(xla::XlaOp input,
                                       absl::Span<int64_t const> output_dims) {
  return xla::BroadcastTo(input, output_dims);
}

absl::Status BroadcastOpsToSame(xla::XlaOp* lhs, xla::XlaOp* rhs) {
  TF_ASSIGN_OR_RETURN(auto lhs_xla_shape, lhs->builder()->GetShape(*lhs));
  TF_ASSIGN_OR_RETURN(auto rhs_xla_shape, rhs->builder()->GetShape(*rhs));
  machina::TensorShape lhs_tf_shape;
  machina::TensorShape rhs_tf_shape;
  TF_RETURN_IF_ERROR(XLAShapeToTensorShape(lhs_xla_shape, &lhs_tf_shape));
  TF_RETURN_IF_ERROR(XLAShapeToTensorShape(rhs_xla_shape, &rhs_tf_shape));
  if (!lhs_tf_shape.IsSameSize(rhs_tf_shape)) {
    machina::BCast bcast(machina::BCast::FromShape(lhs_tf_shape),
                            machina::BCast::FromShape(rhs_tf_shape));
    if (!bcast.IsValid()) {
      return machina::errors::InvalidArgument(
          "Dimensions cannot be made to match through broadcasting");
    }
    TF_ASSIGN_OR_RETURN(*lhs, xla::BroadcastTo(*lhs, bcast.output_shape()));
    TF_ASSIGN_OR_RETURN(*rhs, xla::BroadcastTo(*rhs, bcast.output_shape()));
  }
  return absl::OkStatus();
}

}  // namespace machina
