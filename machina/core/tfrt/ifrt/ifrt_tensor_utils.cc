/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, July 12, 2025.
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

#include "machina/core/tfrt/ifrt/ifrt_tensor_utils.h"

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "machina/compiler/tf2xla/type_util.h"
#include "machina/xla/python/ifrt/dtype.h"
#include "machina/xla/python/ifrt/shape.h"
#include "machina/xla/python/pjrt_ifrt/pjrt_dtype.h"
#include "machina/xla/tsl/platform/errors.h"
#include "machina/xla/tsl/platform/statusor.h"
#include "machina/xla/xla_data.pb.h"
#include "machina/core/framework/tensor_shape.h"
#include "machina/core/framework/tensor_shape.pb.h"
#include "machina/core/framework/types.pb.h"

namespace machina {
namespace ifrt_serving {

absl::StatusOr<machina::DataType> ToTensorDataType(
    xla::ifrt::DType ifrt_dtype) {
  if (ifrt_dtype.kind() == xla::ifrt::DType::kString) {
    return machina::DataType::DT_STRING;
  }
  TF_ASSIGN_OR_RETURN(xla::PrimitiveType primitive_type,
                      xla::ifrt::ToPrimitiveType(ifrt_dtype));
  return machina::EncodePrimitiveTypeAsDataType(primitive_type);
}

absl::StatusOr<xla::ifrt::DType> ToIfrtDType(
    machina::DataType tensor_dtype) {
  if (tensor_dtype == machina::DataType::DT_STRING) {
    return xla::ifrt::DType(xla::ifrt::DType::kString);
  }
  xla::PrimitiveType primitive_type;
  TF_RETURN_IF_ERROR(
      machina::DataTypeToPrimitiveType(tensor_dtype, &primitive_type));
  return xla::ifrt::ToDType(primitive_type);
}

xla::ifrt::Shape ToIfrtShape(const machina::TensorShape& shape) {
  return xla::ifrt::Shape(shape.dim_sizes());
}

machina::TensorShape ToTensorShape(const xla::ifrt::Shape& shape) {
  return machina::TensorShape(shape.dims());
}
}  // namespace ifrt_serving
}  // namespace machina
