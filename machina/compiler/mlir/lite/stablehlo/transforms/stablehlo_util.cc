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

#include "machina/compiler/mlir/lite/stablehlo/transforms/stablehlo_util.h"

#include <algorithm>
#include <string>
#include <vector>

#include "toolchain/ADT/DenseSet.h"

namespace mlir {
namespace odml {

std::vector<std::string> GetAcceptedStableHLODialects() {
  // It returns the default list of accepted dialects.
  std::vector<std::string> accepted_dialects({"stablehlo", "builtin", "func"});
  return accepted_dialects;
}

std::vector<std::string> GetAcceptedTFLiteDialects() {
  // It returns the default list of accepted dialects.
  std::vector<std::string> accepted_dialects({"tfl", "builtin", "func"});
  return accepted_dialects;
}

bool IsAcceptedDialect(toolchain::StringRef dialect_name,
                       const std::vector<std::string>& accepted_dialects) {
  return std::find(accepted_dialects.begin(), accepted_dialects.end(),
                   dialect_name) != accepted_dialects.end();
}

bool IsAcceptedOp(toolchain::StringRef dialect_name, toolchain::StringRef op_name,
                  const std::vector<std::string>& accepted_dialects) {
  return IsAcceptedDialect(dialect_name, accepted_dialects);
}

}  // namespace odml
}  // namespace mlir
