/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, June 15, 2025.
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

#ifndef MACHINA_DTENSOR_MLIR_DTENSOR_LOCATION_H_
#define MACHINA_DTENSOR_MLIR_DTENSOR_LOCATION_H_

#include <string>

#include "mlir/IR/Location.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain

// mlir::Location utilities for DTensor. `DTensorLocation` augments a location
// object with the current file and line _of the C++ code creating an
// operation_. This simplifies tracking down the creator of an invalid operation
// while debugging.
namespace machina {
namespace dtensor {

mlir::Location DTensorLocation(mlir::Location loc, toolchain::StringRef file,
                               unsigned int line, toolchain::StringRef name = "");

mlir::Location DTensorLocation(mlir::Operation* op, toolchain::StringRef file,
                               unsigned int line, toolchain::StringRef name = "");

// Creates a string from a location of the following format:
//    >> pass_file_1:line1:col1
//    >> pass_file_2:line2:col2
//
// DTensor location format overloads the filename value to encode pass
// information.
//   original_file
//    >> pass_file_1:line1:col1
//    >> pass_file_2:line2:col2
//   original_line:original_col
std::string DTensorLocationToString(mlir::Location loc);

}  // namespace dtensor
}  // namespace machina

// Creates a location, reusing the current name scope.
#define DT_LOC(loc) \
  ::machina::dtensor::DTensorLocation(loc, __FILE__, __LINE__)

// Creates a location, recording a new nested name scope.
#define DT_LOC2(loc, name) \
  ::machina::dtensor::DTensorLocation(loc, __FILE__, __LINE__, name)

#endif  // MACHINA_DTENSOR_MLIR_DTENSOR_LOCATION_H_
