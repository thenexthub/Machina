/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Friday, August 8, 2025.
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

#include "machina/xla/service/gpu/model/experimental/symbolic_tiled_hlo.h"

#include <sstream>
#include <string>

#include "absl/strings/string_view.h"
#include "machina/xla/service/gpu/model/experimental/symbolic_tile.h"

namespace xla::gpu::experimental {

std::string SymbolicTiledHloInstruction::ToString(
    absl::string_view field_separator) const {
  std::stringstream ss;
  ss << "hlo: " << hlo_->ToString() << field_separator;
  ss << "tile: " << symbolic_tile().ToString() << field_separator;
  return ss.str();
}

}  // namespace xla::gpu::experimental
