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

#include "machina/xla/service/gpu/model/tiled_hlo_computation.h"

#include <sstream>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "machina/xla/hlo/ir/hlo_opcode.h"
#include "machina/xla/service/gpu/model/tiled_hlo_instruction.h"
#include "machina/xla/service/name_uniquer.h"
#include "machina/xla/util.h"

namespace xla {
namespace gpu {

std::string TiledHloComputation::ToString() const {
  std::stringstream ss;
  NameUniquer name_uniquer("_");
  absl::flat_hash_map<const TiledHloInstruction*, std::string> tile_names;

  for (const auto* tiled_hlo : instructions()) {
    std::string tile_name = name_uniquer.GetUniqueName(
        absl::StrCat(tiled_hlo->hlo()->name(), ".tile_0"));
    tile_names[tiled_hlo] = tile_name;

    absl::InlinedVector<std::string, 4> operand_names;
    for (const auto& operand : tiled_hlo->operands()) {
      operand_names.push_back(tile_names.at(operand));
    }

    ss << tile_name << " = " << HloOpcodeString(tiled_hlo->hlo()->opcode())
       << "(" << absl::StrJoin(operand_names, ", ") << ")\n";

    ss << tiled_hlo->ToString() << "\n";
  }
  return ss.str();
}

}  // namespace gpu
}  // namespace xla
