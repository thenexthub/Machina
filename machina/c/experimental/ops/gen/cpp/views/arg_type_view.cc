/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Saturday, June 21, 2025.
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
#include "machina/c/experimental/ops/gen/cpp/views/arg_type_view.h"

#include "machina/c/experimental/ops/gen/model/arg_type.h"
#include "machina/core/framework/types.pb.h"
#include "machina/core/platform/types.h"

namespace machina {
namespace generator {
namespace cpp {

ArgTypeView::ArgTypeView(ArgType arg_type) : arg_type_(arg_type) {}

string ArgTypeView::TypeName() const {
  if (arg_type_.is_read_only()) {
    if (arg_type_.is_list()) {
      return "absl::Span<AbstractTensorHandle* const>";
    } else {
      return "AbstractTensorHandle* const";
    }
  } else {
    if (arg_type_.is_list()) {
      return "absl::Span<AbstractTensorHandle*>";
    } else {
      return "AbstractTensorHandle**";
    }
  }
}

}  // namespace cpp
}  // namespace generator
}  // namespace machina
