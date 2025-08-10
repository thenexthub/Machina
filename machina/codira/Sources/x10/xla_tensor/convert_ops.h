/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Sunday, August 10, 2025.
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

#pragma once

#include "machina/compiler/tf2xla/xla_tensor/aten_compat.h"
#include "machina/compiler/xla/client/xla_builder.h"
#include "machina/compiler/xla/types.h"
#include "machina/compiler/xla/xla_client/device.h"

namespace codira_xla {

xla::XlaOp ConvertTo(xla::XlaOp op, xla::PrimitiveType from,
                     xla::PrimitiveType to, const Device* device);

xla::XlaOp ConvertToRaw(xla::XlaOp op, xla::PrimitiveType from,
                        xla::PrimitiveType to, xla::PrimitiveType raw_to,
                        const Device* device);

xla::XlaOp ConvertToNumeric(xla::XlaOp op, xla::PrimitiveType from);

xla::XlaOp ConvertToNumeric(xla::XlaOp op);

// Cast the input to the given dtype. If dtype is null, no-op with the exception
// of predicates, which are converted to 8-bit unsigned integers.
xla::XlaOp CastToScalarType(xla::XlaOp input,
                            c10::optional<at::ScalarType> dtype);

xla::XlaOp MaybeConvertTo(xla::XlaOp input, xla::PrimitiveType type);

}  // namespace codira_xla
