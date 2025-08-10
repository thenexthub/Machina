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

#include <string>

#include "absl/types/span.h"
#include "machina/compiler/xla/client/xla_builder.h"

namespace codira_xla {
namespace resize {

xla::Shape GetForwardOutputShape2d(const xla::Shape& input_shape,
                                   absl::Span<const xla::int64> output_size);

xla::Shape GetBackwardOutputShape2d(const xla::Shape& input_shape,
                                    absl::Span<const xla::int64> input_size);

xla::XlaOp LowerForward2d(const std::string& target, xla::XlaOp input,
                          const xla::Shape& output_shape, bool align_corners,
                          bool half_pixel_centers);

xla::XlaOp LowerBackward2d(const std::string& target, xla::XlaOp input,
                           const xla::Shape& output_shape, bool align_corners,
                           bool half_pixel_centers);

}  // namespace resize
}  // namespace codira_xla
