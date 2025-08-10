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

#include "machina/compiler/mlir/machina/transforms/constant_fold_utils.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "toolchain/ADT/ArrayRef.h"
#include "toolchain/ADT/SmallVector.h"
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_ops.h"
#include "machina/compiler/mlir/machina/ir/tf_traits.h"
#include "machina/compiler/mlir/machina/utils/convert_tensor.h"
#include "machina/compiler/mlir/machina/utils/translate_utils.h"
#include "machina/core/framework/function.pb.h"
#include "machina/core/framework/node_def.pb.h"
#include "machina/core/tfrt/fallback/fallback_state.h"
#include "machina/core/tfrt/fallback/op_kernel_runner.h"

namespace mlir {
namespace TF {

using machina::tfrt_stub::FallbackState;
using machina::tfrt_stub::OpKernelRunner;

static bool IsOk(const absl::Status& s) {
  if (s.ok()) return true;
  VLOG(2) << s.message();
  return false;
}

#define RETURN_FAILURE_IF_ERROR(expr) \
  if (!IsOk(expr)) {                  \
    return mlir::failure();           \
  }

bool CanBeFolded(Operation* inst) {
  // Instructions with side effects should not be constant folded to preserve
  // the original semantics. Ops that have no side effect and zero results but
  // could be folded should have a custom folder instead of relying on the
  // TensorFlow folding hook.
  if (inst == nullptr || inst->getNumResults() == 0 ||
      inst->hasTrait<OpTrait::TF::NoConstantFold>() ||
      inst->getNumRegions() != 0 || !isMemoryEffectFree(inst)) {
    return false;
  }

  // If any of the result types are variants, don't try to constant fold them.
  // This creates opaque variant constants which lose information and would
  // require "raising" later.
  for (const Type type : inst->getResultTypes()) {
    if (const TensorType tensor_type = mlir::dyn_cast<TensorType>(type)) {
      if (mlir::isa<VariantType>(tensor_type.getElementType())) {
        return false;
      }
    }
  }

  // Operations that execute function calls shouldn't be constant folded.
  if (toolchain::isa<TF::WhileOp, TF::CaseOp, TF::IfOp, CallOpInterface>(inst)) {
    return false;
  }

  return true;
}

static const FallbackState& GetDefaultFallbackState() {
  static const auto* const fallback_state = []() {
    machina::SessionOptions session_options;
    machina::FunctionDefLibrary fdef_lib;
    auto fallback_state =
        FallbackState::CreateWithCpuDevice(session_options, fdef_lib).value();
    return fallback_state.release();
  }();

  return *fallback_state;
}

static std::function<void(std::function<void()>)>* GetDefaultRunner() {
  static auto* const default_runner =
      new std::function<void(std::function<void()>)>(
          [](const std::function<void()>& f) { f(); });
  return default_runner;
}

LogicalResult EvaluateOperation(Operation* inst,
                                toolchain::ArrayRef<ElementsAttr> operands,
                                toolchain::SmallVector<Attribute>& results) {
  // If any operand is nullptr returns true for a failure.
  // TODO(b/120678030): remove this constraint if we find operators can be
  // evaluated with some unknown operands.
  if (std::any_of(operands.begin(), operands.end(),
                  [](Attribute operand) { return !operand; })) {
    VLOG(1) << "Can't evaluate since not all operands are constant.";
    return failure();
  }

  // Builds TF operation and sets all the attributes.
  std::string node_name = "unnamed";
  if (const StringAttr attr = inst->getAttrOfType<StringAttr>("name")) {
    node_name = std::string(attr.getValue());
  }
  absl::StatusOr<std::unique_ptr<machina::NodeDef>> node_def =
      machina::ConvertTFDialectOpToNodeDef(
          inst, node_name.c_str(), /*ignore_unregistered_attrs=*/true);
  RETURN_FAILURE_IF_ERROR(node_def.status());

  const FallbackState& fallback_state = GetDefaultFallbackState();

  // Explicitly set device to Host CPU instead of the device present in device
  // attribute of the MLIR op. The assigned device might be remote, not
  // available during compilation or compilation only device for on demand
  // execution which may create a recursion if used for constant folding.
  std::string host_cpu = machina::DeviceNameUtils::FullName(
      /*job=*/"localhost", /*replica=*/0, /*task=*/0, /*type=*/"CPU", /*id=*/0);

  absl::StatusOr<OpKernelRunner> runner = OpKernelRunner::Create(
      node_def->get()->op(), node_def->get()->name(), host_cpu, operands.size(),
      [&](machina::AttrValueMap* attr_value_map) {
        *attr_value_map = node_def->get()->attr();
        return absl::OkStatus();
      },
      fallback_state.device_manager(),
      fallback_state.process_function_library_runtime());
  RETURN_FAILURE_IF_ERROR(runner.status());

  VLOG(1) << "Start to evaluate node: " << node_def->get()->DebugString();

  std::vector<machina::Tensor> inputs;

  // Adds inputs to the TF operation.
  for (const ElementsAttr& operand : operands) {
    machina::Tensor tensor;
    RETURN_FAILURE_IF_ERROR(machina::ConvertToTensor(operand, &tensor));
    inputs.push_back(std::move(tensor));
  }

  std::vector<machina::TensorValue> input_values;
  for (machina::Tensor& tensor : inputs) {
    input_values.emplace_back();
    input_values.back().tensor = &tensor;
  }

  machina::OpKernelContext::Params params;
  params.inputs = input_values;
  params.device = runner->device();
  params.op_kernel = runner->op_kernel();

  // Still use original device's resource_manager.
  params.resource_manager = runner->resource_manager();
  params.input_alloc_attrs = runner->input_alloc_attrs();
  params.output_attr_array = runner->output_alloc_attrs().data();

  // Following two parameters are used to support executing tf.data via
  // fallback.
  params.function_library = runner->function_library_runtime();
  params.runner = GetDefaultRunner();

  // Executes the TF operation.
  machina::OpKernelContext op_kernel_context(&params);
  runner->Run(&op_kernel_context);
  RETURN_FAILURE_IF_ERROR(op_kernel_context.status());

  // Converts the outputs to MLIR attributes.
  Builder builder(inst->getContext());

  for (int i = 0; i < op_kernel_context.num_outputs(); ++i) {
    DCHECK(op_kernel_context.mutable_output(i));
    absl::StatusOr<ElementsAttr> result_attr = machina::ConvertTensor(
        *op_kernel_context.mutable_output(i), &builder);
    RETURN_FAILURE_IF_ERROR(result_attr.status());
    results.push_back(result_attr.value());
  }

  VLOG(1) << "Evaluate node " << node_name << " successfully!";

  return success();
}

}  // namespace TF
}  // namespace mlir
