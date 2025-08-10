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

#include "machina/dtensor/mlir/sparse_expander.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "machina/core/framework/registration/registration.h"
#include "machina/core/platform/status.h"
#include "machina/dtensor/mlir/op_utils.h"
#include "machina/dtensor/mlir/sparse_expander_common.h"

namespace machina {
namespace dtensor {

// static
SparseExpanderRegistry* SparseExpanderRegistry::Global() {
  static SparseExpanderRegistry* registry = new SparseExpanderRegistry();
  return registry;
}

SparseExpanderBase* SparseExpanderRegistry::GetSparseExpansionFnForOp(
    mlir::Operation* op) {
  auto key = OpName(op);
  auto fn = op_to_sparse_expansion_fn_map_.find(key);
  if (fn == op_to_sparse_expansion_fn_map_.end()) return nullptr;
  return fn->second.get();
}

InitOnStartupMarker SparseExpanderRegistry::RegisterSparseExpansionFn(
    std::string opName, std::unique_ptr<SparseExpanderBase> prop) {
  CHECK(op_to_sparse_expansion_fn_map_  // Crash ok
            .insert_or_assign(opName, std::move(prop))
            .second);
  return {};
}

absl::Status RunSparseExpansion(mlir::Operation* op, mlir::Operation** output) {
  // Only expand if there are any SparseTensor inputs.
  if (HasAnySparseInput(op)) {
    SparseExpanderBase* expander =
        SparseExpanderRegistry::Global()->GetSparseExpansionFnForOp(op);
    if (expander != nullptr) {
      auto expanded_op = expander->ExpandOp(op);
      if (expanded_op.ok()) *output = expanded_op.value();
      return expanded_op.status();
    } else {
      VLOG(1) << "No sparse expansion found for " << OpName(op) << "\n";
      *output = op;
    }
  } else {  // If there is no SparseTensor inputs then just return the op.
    *output = op;
  }
  return absl::OkStatus();
}

}  // namespace dtensor
}  // namespace machina
