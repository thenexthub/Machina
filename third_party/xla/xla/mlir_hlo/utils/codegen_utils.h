/* Copyright 2021 The OpenXLA Authors.

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

#ifndef MLIR_HLO_UTILS_CODEGEN_UTILS_H
#define MLIR_HLO_UTILS_CODEGEN_UTILS_H

#include "toolchain/ADT/SmallVector.h"
#include "toolchain/ADT/StringRef.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/MLIRContext.h"  // TF:toolchain-project

namespace mlir {
class Value;
class ValueRange;
class OpBuilder;
class Location;
class Operation;
namespace codegen_utils {

Value emitNumElementsComputation(OpBuilder& b, Location loc, Operation* op);
Value emitNumElementsComputation(OpBuilder& b, Location loc, Value memref);

toolchain::SmallVector<Value> calcMultiDimIndex(OpBuilder& b, Location loc,
                                           Value linearIndex, Value memref);

toolchain::SmallVector<Value> calcMultiDimIndex(OpBuilder& b, Location loc,
                                           Value linearIndex,
                                           toolchain::ArrayRef<Value> shape);

}  // namespace codegen_utils
}  // namespace mlir

#endif  // MLIR_HLO_UTILS_CODEGEN_UTILS_H
