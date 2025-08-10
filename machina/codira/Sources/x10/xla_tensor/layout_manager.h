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

#include "absl/types/span.h"
#include "machina/compiler/xla/shape.h"
#include "machina/compiler/xla/types.h"
#include "machina/compiler/xla/xla_client/device.h"

namespace codira_xla {

// Creates a minor-to-major layout from given dimensions. The dynamic_dimensions
// slice should be either empty, or of the same size as dimensions.
xla::Shape MakeSwiftTensorLayout(absl::Span<const xla::int64> dimensions,
                                 absl::Span<const bool> dynamic_dimensions,
                                 xla::PrimitiveType type);

// Create an XLA shape with the given dimensions and type, suitable to be used
// in the specified device type. The type of device can affect the choice of the
// XLA layout. The dynamic_dimensions slice should be either empty, or of the
// same size as dimensions.
xla::Shape MakeArrayShapeFromDimensions(
    absl::Span<const xla::int64> dimensions,
    absl::Span<const bool> dynamic_dimensions, xla::PrimitiveType type,
    DeviceType device_type);

}  // namespace codira_xla
