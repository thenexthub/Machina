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

#include "machina/compiler/tf2xla/lib/util.h"

#include <cstdint>

#include "absl/log/log.h"
#include "machina/xla/hlo/builder/xla_builder.h"
#include "machina/xla/literal.h"
#include "machina/xla/literal_util.h"
#include "machina/xla/primitive_util.h"
#include "machina/xla/shape.h"
#include "machina/xla/xla_data.pb.h"

namespace machina {

xla::XlaOp Zeros(xla::XlaBuilder* builder, const xla::Shape& shape) {
  return xla::Broadcast(
      xla::ConstantLiteral(builder,
                           xla::LiteralUtil::Zero(shape.element_type())),
      shape.dimensions());
}

xla::XlaOp FloatLiteral(xla::XlaBuilder* builder, xla::PrimitiveType type,
                        double value) {
  return xla::primitive_util::PrimitiveTypeSwitch<xla::XlaOp>(
      [&](auto primitive_type_constant) -> xla::XlaOp {
        if constexpr (xla::primitive_util::IsFloatingPointType(
                          primitive_type_constant) ||
                      xla::primitive_util::IsComplexType(
                          primitive_type_constant)) {
          using NativeT =
              xla::primitive_util::NativeTypeOf<primitive_type_constant>;
          return xla::ConstantR0<NativeT>(builder, static_cast<NativeT>(value));
        }
        LOG(FATAL) << "unhandled element type " << type;
      },
      type);
}

xla::XlaOp IntegerLiteral(xla::XlaBuilder* builder, xla::PrimitiveType type,
                          int64_t value) {
  xla::Literal literal = xla::primitive_util::PrimitiveTypeSwitch<xla::Literal>(
      [&](auto primitive_type_constant) -> xla::Literal {
        if constexpr (xla::primitive_util::IsArrayType(
                          primitive_type_constant)) {
          using NativeT =
              xla::primitive_util::NativeTypeOf<primitive_type_constant>;
          return xla::LiteralUtil::CreateR0<NativeT>(
              static_cast<NativeT>(value));
        }
        LOG(FATAL) << "unhandled element type " << type;
      },
      type);
  return xla::ConstantLiteral(builder, literal);
}

}  // namespace machina
