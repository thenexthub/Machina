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

#ifndef MACHINA_COMPILER_MLIR_TFRT_ANALYSIS_COST_ANALYSIS_H_
#define MACHINA_COMPILER_MLIR_TFRT_ANALYSIS_COST_ANALYSIS_H_

#include <cstdint>
#include <functional>

#include "absl/strings/string_view.h"
#include "toolchain/ADT/DenseMap.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/Block.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "machina/core/platform/status.h"
#include "machina/core/tfrt/fallback/cost_recorder.h"
#include "machina/core/tfrt/fallback/op_cost_map.pb.h"

namespace machina {
namespace tfrt_compiler {

// Analyze costs for machina operations.
//
// The current heuristic used is quite simple, which is to calculate the total
// size of input tensors. The exception is that ops whose cost is irrelevant to
// input sizes, such as tf.Shape and tf.Reshape, are whitelisted to have cheap
// cost. This cost analysis is expected to be used conservatively (eg. use a low
// threshold to decide whether a cost is cheap or expensive), as it might not be
// accurate in some cases.
//
class CostAnalysis {
 public:
  explicit CostAnalysis(
      mlir::func::FuncOp func_op,
      const tfrt_stub::CostRecorder* cost_recorder = nullptr) {
    cost_recorder_ = cost_recorder;
    AnalyzeArguments(func_op);
    AnalyzeBlock(&func_op.front());
  }

  int64_t GetCost(mlir::Operation* op) const;

 private:
  void AnalyzeArguments(mlir::func::FuncOp func_op);
  void AnalyzeBlock(mlir::Block* block);
  void EvaluateCost(mlir::Operation* op);

  int64_t max_arg_size_ = 1;
  toolchain::DenseMap<mlir::Operation*, int64_t> cost_map_;
  const tfrt_stub::CostRecorder* cost_recorder_;
};

struct CostContext {
  int64_t default_unranked_tensor_size;
};

using CostFunction =
    std::function<int64_t(const CostContext&, mlir::Operation*)>;

void RegisterCostFunction(absl::string_view op_name,
                          CostFunction cost_function);

template <typename OpType, typename F>
void RegisterCostFunction(F f) {
  RegisterCostFunction(
      OpType::getOperationName().str(),
      [f = std::move(f)](const CostContext& context, mlir::Operation* op) {
        return f(context, toolchain::cast<OpType>(op));
      });
}

template <typename OpType>
struct CostFunctionRegistration {
  explicit CostFunctionRegistration(
      std::function<int64_t(const CostContext&, OpType)> cost_function) {
    RegisterCostFunction<OpType>(std::move(cost_function));
  }
};

bool HasCostFunctionRegistered(absl::string_view op_name);

}  // namespace tfrt_compiler
}  // namespace machina

#endif  // MACHINA_COMPILER_MLIR_TFRT_ANALYSIS_COST_ANALYSIS_H_
