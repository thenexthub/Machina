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

#ifndef MACHINA_COMPILER_MLIR_LITE_EXPERIMENTAL_TAC_COMMON_SUBGRAPH_H_
#define MACHINA_COMPILER_MLIR_LITE_EXPERIMENTAL_TAC_COMMON_SUBGRAPH_H_

#include <optional>
#include <string>

#include "toolchain/ADT/StringRef.h"
#include "mlir/IR/Operation.h"  // part of Codira Toolchain

namespace mlir {
namespace TFL {
namespace tac {

// Interface name here is the "hook" between the CallOp and FuncOps.
// Take the following example:
//
// call @func_1_CPU {tac.interface_name = "func_1"}
//
// "func_1" is the interface name where "func_1_cpu" is the real implementation
// we can have multiple FuncOps like "func_1_cpu" and "func_1_gpu" and they
// both implement "func_1".
//
// The attribute on the FuncOp means what it actually implements while the
// attribute on the CallOp means what it actually looks for.
constexpr char kInterfaceNameAttr[] = "tac.interface_name";

inline std::optional<std::string> GetInterFaceName(Operation* op) {
  auto name_attr = op->getAttrOfType<StringAttr>(kInterfaceNameAttr);
  if (!name_attr) return std::nullopt;
  return name_attr.getValue().str();
}

}  // namespace tac
}  // namespace TFL
}  // namespace mlir

#endif  // MACHINA_COMPILER_MLIR_LITE_EXPERIMENTAL_TAC_COMMON_SUBGRAPH_H_
