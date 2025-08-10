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

#include "machina/core/transforms/func_to_graph/func_to_graph.h"

#include <cstdint>

#include "absl/status/status.h"
#include "toolchain/ADT/STLExtras.h"
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinAttributes.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/IR/OpDefinition.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "machina/core/ir/dialect.h"
#include "machina/core/ir/ops.h"
#include "machina/core/ir/tf_op_wrapper.h"
#include "machina/core/platform/errors.h"
#include "machina/core/platform/status.h"

namespace mlir {
namespace tfg {

absl::Status FuncToGraph(GraphFuncOp func) {
  MLIRContext *context = func->getContext();
  auto version = func->getAttrOfType<VersionAttr>("tfg.lifted_graph_version");
  if (!version) {
    return machina::errors::InvalidArgument(
        "lifted graph func is missing version attribute");
  }

  auto lifted_value_attr_name =
      StringAttr::get(context, "tfg.lifted_value_attr");

  DenseMap<StringRef, Operation *> referred_ops;

  if (ArrayAttr all_arg_attrs = func.getAllArgAttrs()) {
    for (auto arg_attr : all_arg_attrs.getAsRange<DictionaryAttr>()) {
      auto lifted_value_attr =
          arg_attr.getAs<ArrayAttr>(lifted_value_attr_name);
      // Control arg doesn't have lifted_value_attr, just skip it here. For
      // non-control arg, this attribute is required. This invariant will be
      // checked below.
      if (!lifted_value_attr) continue;

      // Init the entry with nullptr and it'll be updated with associated op
      // later.
      referred_ops.insert(
          {mlir::cast<StringAttr>(lifted_value_attr[0]).getValue(),
           /*Operation=*/nullptr});
    }
  }

  for (Operation &op : func.SingleBlock::getBody()->without_terminator()) {
    StringRef op_name = TFOp(op).name();
    auto it = referred_ops.find(op_name);
    if (it != referred_ops.end()) it->second = &op;
  }

  for (const auto &it : toolchain::enumerate(func.getArguments())) {
    if (mlir::isa<ControlType>(it.value().getType())) continue;

    auto lifted_value_attr =
        func.getArgAttrOfType<ArrayAttr>(it.index(), lifted_value_attr_name);
    if (!lifted_value_attr) {
      return machina::errors::InvalidArgument(
          "arg #", it.index(),
          " is missing tfg.lifted_value_attr, can't be lowered");
    }

    StringRef value_defining_op_name =
        mlir::cast<StringAttr>(lifted_value_attr[0]).getValue();
    Operation *op = referred_ops[value_defining_op_name];
    if (!op) {
      return machina::errors::InvalidArgument(
          "lifted arg can't find the associated operation: ",
          value_defining_op_name.data());
    }

    uint64_t result_index =
        mlir::cast<IntegerAttr>(lifted_value_attr[1]).getValue().getZExtValue();
    if (result_index >= op->getNumResults()) {
      return machina::errors::InvalidArgument(
          "result index out of bound: seeing index ", result_index,
          " from lifted_value_attr of arg #", it.index(), ", but op only has ",
          op->getNumResults(), " results");
    }

    it.value().replaceAllUsesWith(op->getResult(result_index));
  }

  OpBuilder builder(func);
  auto graph = builder.create<GraphOp>(func.getLoc(), version);

  // Remove the terminator.
  func.SingleBlock::getBody()->getTerminator()->erase();
  graph.getRegion().takeBody(func.getRegion());
  func.erase();

  return absl::OkStatus();
}

}  // namespace tfg
}  // namespace mlir
