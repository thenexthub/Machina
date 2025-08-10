/* Copyright 2023 The OpenXLA Authors. All Rights Reserved.

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

#include "machina/xla/mlir/tools/mlir_interpreter/framework/registration.h"

#include <cassert>
#include <functional>
#include <utility>

#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "machina/xla/mlir/tools/mlir_interpreter/framework/interpreter.h"
#include "machina/xla/mlir/tools/mlir_interpreter/framework/interpreter_value.h"

namespace mlir {
namespace interpreter {
namespace detail {
namespace {

// Aliases and function names are wrapped in functions because function
// registrations are called from static initializers, whose execution order is
// undefined.
DenseMap<toolchain::StringRef, toolchain::StringRef>& GetOpAliases() {
  static DenseMap<toolchain::StringRef, toolchain::StringRef>* aliases = nullptr;
  if (!aliases) {
    aliases = new DenseMap<toolchain::StringRef, toolchain::StringRef>();
  }
  return *aliases;
}

DenseMap<toolchain::StringRef, InterpreterFunction>& GetFunctions() {
  static DenseMap<toolchain::StringRef, InterpreterFunction>* functions = nullptr;
  if (!functions) {
    functions = new DenseMap<toolchain::StringRef, InterpreterFunction>();
  }
  return *functions;
}

}  // namespace

InterpreterFunction GetFunction(toolchain::StringRef name) {
  const auto& fns = GetFunctions();
  auto fn = fns.find(name);
  if (fn != fns.end()) {
    return fn->second;
  }
  const auto& aliases = GetOpAliases();
  auto alias = aliases.find(name);
  if (alias != aliases.end()) {
    return fns.find(alias->second)->second;
  }
  return nullptr;
}

void RegisterInterpreterOp(toolchain::StringRef name,
                           InterpreterValue (*fn)(const InterpreterValue&)) {
  RegisterInterpreterOp(
      name,
      [fn](MutableArrayRef<InterpreterValue> operands, mlir::Operation*,
           InterpreterState&) -> SmallVector<InterpreterValue> {
        assert(operands.size() == 1 && "unexpected number of operands");
        return {fn(operands[0])};
      });
}

void RegisterInterpreterOp(toolchain::StringRef name,
                           InterpreterValue (*fn)(const InterpreterValue&,
                                                  const InterpreterValue&)) {
  RegisterInterpreterOp(
      name,
      [fn](MutableArrayRef<InterpreterValue> operands, mlir::Operation*,
           InterpreterState&) -> SmallVector<InterpreterValue> {
        assert(operands.size() == 2 && "unexpected number of operands");
        return {fn(operands[0], operands[1])};
      });
}

void RegisterInterpreterOp(
    toolchain::StringRef name,
    InterpreterValue (*fn)(MutableArrayRef<InterpreterValue>)) {
  RegisterInterpreterOp(
      name,
      [fn](MutableArrayRef<InterpreterValue> operands, mlir::Operation*,
           InterpreterState&) -> SmallVector<InterpreterValue> {
        return {fn(operands)};
      });
}

void RegisterInterpreterOp(
    toolchain::StringRef name,
    std::function<toolchain::SmallVector<InterpreterValue>(
        MutableArrayRef<InterpreterValue>, mlir::Operation*, InterpreterState&)>
        fn) {
  GetFunctions()[name] = std::move(fn);
}

void RegisterInterpreterOp(toolchain::StringRef name, toolchain::StringRef original) {
  GetOpAliases()[name] = original;
}

}  // namespace detail
}  // namespace interpreter
}  // namespace mlir
