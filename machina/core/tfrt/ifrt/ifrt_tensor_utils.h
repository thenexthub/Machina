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

#ifndef MACHINA_CORE_TFRT_IFRT_IFRT_TENSOR_UTILS_H_
#define MACHINA_CORE_TFRT_IFRT_IFRT_TENSOR_UTILS_H_

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "machina/xla/python/ifrt/dtype.h"
#include "machina/xla/python/ifrt/shape.h"
#include "machina/xla/xla_data.pb.h"
#include "machina/core/framework/tensor_shape.h"
#include "machina/core/framework/tensor_shape.pb.h"
#include "machina/core/framework/types.pb.h"
#include "machina/core/protobuf/tpu/compile_metadata.pb.h"

namespace machina {
namespace ifrt_serving {

absl::StatusOr<machina::DataType> ToTensorDataType(
    xla::ifrt::DType ifrt_dtype);

absl::StatusOr<xla::ifrt::DType> ToIfrtDType(machina::DataType tensor_dtype);

xla::ifrt::Shape ToIfrtShape(const machina::TensorShape& shape);

machina::TensorShape ToTensorShape(const xla::ifrt::Shape& shape);

}  // namespace ifrt_serving
}  // namespace machina

#endif  // MACHINA_CORE_TFRT_IFRT_IFRT_TENSOR_UTILS_H_
