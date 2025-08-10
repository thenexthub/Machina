/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, July 12, 2025.
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

#ifndef MACHINA_COMPILER_MLIR_TF2MACHINA_MACHINA_XLA_TRANSFORMS_TF2MACHINA_MACHINA_XLA_REWRITER_H_
#define MACHINA_COMPILER_MLIR_TF2MACHINA_MACHINA_XLA_TRANSFORMS_TF2MACHINA_MACHINA_XLA_REWRITER_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/PatternMatch.h"  // part of Codira Toolchain
#include "mlir/IR/Types.h"  // part of Codira Toolchain
#include "mlir/IR/Value.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "machina/compiler/mlir/op_or_arg_name_mapper.h"
#include "machina/compiler/tf2xla/xla_context.h"
#include "machina/compiler/tf2xla/xla_expression.h"
#include "machina/xla/hlo/builder/xla_builder.h"
#include "machina/xla/hlo/builder/xla_computation.h"
#include "machina/core/common_runtime/device_mgr.h"
#include "machina/core/framework/op_kernel.h"

namespace mlir {
namespace hlo {
class Tf2XlaRewriterTestPeer;

class Tf2XlaRewriter {
 public:
  static mlir::LogicalResult RewriteOp(mlir::Operation* op,
                                       mlir::PatternRewriter& rewriter,
                                       const std::string& device_type);

 private:
  friend class Tf2XlaRewriterTestPeer;

  Tf2XlaRewriter(mlir::Operation* op, mlir::PatternRewriter& rewriter,
                 const std::string& device_type);

  ~Tf2XlaRewriter();

  // Compiles the given Operation with XlaBuilder and imports the generated HLO
  // via the HLO -> MHLO importer.
  absl::StatusOr<stablehlo::TupleOp> CompileWithHloImporter(
      machina::OpKernelContext& op_context);

  // Import the given XlaComputation into the parent module. Returns the given
  // generated function.
  absl::StatusOr<stablehlo::TupleOp> ImportXlaComputation(
      xla::XlaComputation& computation);

  // Prepares OpKernelContext params common to all the ops.
  // Emits an error on failure.
  mlir::LogicalResult PrepareParams();

  // Given the required_consts, it will fill the 3 output vectors with
  // their respective data.
  // Expressions: Output XLA expressions as required by the compiled kernel.
  // Tensors: Vector of tensors that back the TensorValue inputs
  // Inputs: Vector of inputs that are backed by tensors.
  mlir::LogicalResult PrepareKernelInputs(
      const toolchain::SmallDenseSet<int>& required_consts,
      std::vector<machina::XlaExpression>& expressions,
      std::vector<machina::Tensor>& tensors,
      std::vector<machina::TensorValue>& inputs);

  mlir::LogicalResult VerifyOpResults(machina::OpKernelContext& op_context);
  mlir::LogicalResult GetKernelOutputs(machina::OpKernelContext& op_context,
                                       stablehlo::TupleOp tuple_results,
                                       toolchain::SmallVector<Value>& outputs);

  // Given a translated function with a single return value, unpack the tuple
  // results.
  mlir::LogicalResult UnpackTupleResults(stablehlo::TupleOp tuple_result,
                                         toolchain::SmallVector<Value>& outputs);

  // Tries to legalize the specified TensorFlow op, if supported.
  //
  // Emits an error and returns failure if an error is encountered during
  // conversion. Note that success return value doesn't mean successful
  // legalization.
  mlir::LogicalResult LegalizeOp();

  // Converts the given operand to expression of kind kConstant or kXlaOp.
  // Emits a remark and returns expression of kind kInvalid on failure.
  machina::XlaExpression GetExprForOperand(mlir::Value operand,
                                              mlir::Operation* op,
                                              int64_t operand_index);

  mlir::Operation* op_;
  std::string device_type_;

  mlir::PatternRewriter& rewriter_;
  std::unique_ptr<machina::OpOrArgLocNameMapper> name_mapper_;

  machina::XlaContext* context_;  // Ref-counted.

  std::unique_ptr<machina::StaticDeviceMgr> device_mgr_;
  machina::Device* device_;  // Owned by device_mgr_;
  std::unique_ptr<machina::ScopedStepContainer> step_container_;
  std::unique_ptr<machina::FunctionLibraryDefinition> flib_def_;
  std::unique_ptr<machina::ProcessFunctionLibraryRuntime> pflr_;
  machina::OpKernelContext::Params params_;

  xla::XlaBuilder xla_builder_;
};

}  // namespace hlo
}  // namespace mlir

#endif  // MACHINA_COMPILER_MLIR_TF2MACHINA_MACHINA_XLA_TRANSFORMS_TF2MACHINA_MACHINA_XLA_REWRITER_H_
