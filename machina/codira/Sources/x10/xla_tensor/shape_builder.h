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

#include <vector>

#include "absl/types/span.h"
#include "machina/compiler/xla/shape.h"
#include "machina/compiler/xla/types.h"

namespace codira_xla {

class ShapeBuilder {
 public:
  explicit ShapeBuilder(xla::PrimitiveType type) : type_(type) {}

  ShapeBuilder& Add(const xla::Shape& shape, xla::int64 dim);

  ShapeBuilder& Add(const xla::Shape& shape,
                    absl::Span<const xla::int64> dimensions);

  ShapeBuilder& Add(xla::int64 size);

  xla::Shape Build() const;

 private:
  struct ShapeDim {
    const xla::Shape* shape = nullptr;
    xla::int64 dim_or_size = -1;
  };

  xla::PrimitiveType type_;
  std::vector<ShapeDim> dims_;
};

}  // namespace codira_xla
