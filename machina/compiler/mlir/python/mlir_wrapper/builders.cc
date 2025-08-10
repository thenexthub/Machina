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

#include "mlir/IR/Builders.h"  // part of Codira Toolchain

#include <vector>

#include "mlir/IR/BuiltinAttributes.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypes.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/python/mlir_wrapper/mlir_wrapper.h"

void init_builders(py::module& m) {
  py::class_<mlir::Builder>(m, "Builder")
      .def(py::init<mlir::MLIRContext*>())
      .def("getFunctionType",
           [](mlir::Builder& b, std::vector<mlir::Type> inputs,
              std::vector<mlir::Type> outputs) {
             return b.getFunctionType(toolchain::ArrayRef<mlir::Type>(inputs),
                                      toolchain::ArrayRef<mlir::Type>(outputs));
           });
  py::class_<mlir::OpBuilder>(m, "OpBuilder")
      .def(py::init<mlir::MLIRContext*>())
      .def(py::init<mlir::Region&>())
      .def(py::init<mlir::Operation*>())
      .def(py::init<mlir::Block*, mlir::Block::iterator>())
      .def("getUnknownLoc", &mlir::OpBuilder::getUnknownLoc)
      .def("setInsertionPoint",
           py::overload_cast<mlir::Block*, mlir::Block::iterator>(
               &mlir::OpBuilder::setInsertionPoint))
      .def("saveInsertionPoint", &mlir::OpBuilder::saveInsertionPoint)
      .def("restoreInsertionPoint", &mlir::OpBuilder::restoreInsertionPoint)
      .def(
          "create",
          [](mlir::OpBuilder& opb, mlir::OperationState& state) {
            return opb.create(state);
          },
          py::return_value_policy::reference)
      .def("getContext", &mlir::OpBuilder::getContext,
           py::return_value_policy::reference);

  py::class_<mlir::OpBuilder::InsertPoint>(m, "OpBuilder_InsertionPoint")
      .def("getBlock", &mlir::OpBuilder::InsertPoint::getBlock);
}
