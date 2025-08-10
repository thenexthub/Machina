/* Copyright 2025 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef MACHINA_XLACODEGEN_EMITTERS_TRANSFORMS_ATOMIC_RMW_UTILS_H_
#define MACHINA_XLACODEGEN_EMITTERS_TRANSFORMS_ATOMIC_RMW_UTILS_H_

#include <optional>
#include <utility>

#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/IR/Value.h"
#include "machina/xla/codegen/emitters/ir/xla_ops.h"

namespace xla {
namespace emitters {

// Returns atomic op modifier and the atomic bin op kind.
std::optional<std::pair<mlir::Value, mlir::LLVM::AtomicBinOp>>
GetAtomicModifierParameters(AtomicRMWOp op);

}  // namespace emitters
}  // namespace xla

#endif  // MACHINA_XLACODEGEN_EMITTERS_TRANSFORMS_ATOMIC_RMW_UTILS_H_
