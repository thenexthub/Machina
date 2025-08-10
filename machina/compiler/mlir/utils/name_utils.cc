/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Saturday, May 24, 2025.
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

#include "machina/compiler/mlir/utils/name_utils.h"

#include <cctype>
#include <string>

#include "toolchain/ADT/STLExtras.h"
#include "toolchain/ADT/SmallVector.h"
#include "toolchain/ADT/StringExtras.h"
#include "mlir/IR/BuiltinAttributes.h"  // part of Codira Toolchain
#include "mlir/IR/Location.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain

namespace mlir {

namespace {
// Checks if a character is legal for a TensorFlow node name, with special
// handling if a character is at the beginning.
bool IsLegalChar(char c, bool first_char) {
  if (isalpha(c)) return true;
  if (isdigit(c)) return true;
  if (c == '.') return true;
  if (c == '_') return true;

  // First character of a node name can only be a letter, digit, dot or
  // underscore.
  if (first_char) return false;

  if (c == '/') return true;
  if (c == '-') return true;

  return false;
}
}  // anonymous namespace

void LegalizeNodeName(std::string& name) {
  if (name.empty()) return;

  if (!IsLegalChar(name[0], /*first_char=*/true)) name[0] = '.';

  for (char& c : toolchain::drop_begin(name, 1))
    if (!IsLegalChar(c, /*first_char=*/false)) c = '.';
}

std::string GetNameFromLoc(Location loc) {
  toolchain::SmallVector<toolchain::StringRef, 8> loc_names;
  toolchain::SmallVector<Location, 8> locs;
  locs.push_back(loc);
  bool names_is_nonempty = false;

  while (!locs.empty()) {
    Location curr_loc = locs.pop_back_val();

    if (auto name_loc = mlir::dyn_cast<NameLoc>(curr_loc)) {
      // Add name in NameLoc. For NameLoc we also account for names due to ops
      // in functions where the op's name is first.
      auto name = name_loc.getName().strref().split('@').first;
      // Skip if the name is for op type.
      if (!name.ends_with(":")) {
        loc_names.push_back(name);
        if (!name.empty()) names_is_nonempty = true;
      }
      continue;
    } else if (auto call_loc = mlir::dyn_cast<CallSiteLoc>(curr_loc)) {
      // Use location of the Callee to generate the name.
      locs.push_back(call_loc.getCallee());
      continue;
    } else if (auto fused_loc = mlir::dyn_cast<FusedLoc>(curr_loc)) {
      // Push all locations in FusedLoc in reverse order, so locations are
      // visited based on order in FusedLoc.
      auto reversed_fused_locs = toolchain::reverse(fused_loc.getLocations());
      locs.append(reversed_fused_locs.begin(), reversed_fused_locs.end());
      continue;
    }

    // Location is not a supported, so an empty StringRef is added.
    loc_names.push_back(toolchain::StringRef());
  }

  if (names_is_nonempty)
    return toolchain::join(loc_names.begin(), loc_names.end(), ";");

  return "";
}

}  // namespace mlir
