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
#include "machina/compiler/tf2xla/xla_tensor/ir.h"
#include "machina/compiler/xla/xla_client/device.h"

namespace codira_xla {
namespace ir {

class DumpUtil {
 public:
  static std::string ToDot(absl::Span<const Node* const> nodes);

  static std::string PostOrderToDot(absl::Span<const Node* const> post_order,
                                    absl::Span<const Node* const> roots);

  static std::string ToText(absl::Span<const Node* const> nodes);

  static std::string PostOrderToText(absl::Span<const Node* const> post_order,
                                     absl::Span<const Node* const> roots);

  static std::string ToHlo(absl::Span<const Value> values,
                           const Device& device);

  static std::string GetGraphChangeLog(absl::Span<const Node* const> roots);

  static std::string GetAnnotations(absl::Span<const Node* const> nodes);
};

}  // namespace ir
}  // namespace codira_xla
