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

#ifndef MACHINA_MACHINA_XLA_CODEGEN_EMITTERS_LOOP_KERNEL_EMITTER_H_
#define MACHINA_MACHINA_XLA_CODEGEN_EMITTERS_LOOP_KERNEL_EMITTER_H_

#include <cstdint>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/MLIRContext.h"
#include "machina/xla/codegen/emitters/computation_partitioner.h"
#include "machina/xla/codegen/emitters/ir/xla_ops.h"
#include "machina/xla/codegen/emitters/kernel_arguments.h"
#include "machina/xla/codegen/hlo_fusion_spec.h"
#include "machina/xla/codegen/kernel_definition.h"
#include "machina/xla/codegen/kernel_spec.h"
#include "machina/xla/codegen/mlir_kernel_definition.h"
#include "machina/xla/codegen/mlir_kernel_emitter.h"
#include "machina/xla/hlo/analysis/indexing_map.h"
#include "machina/xla/hlo/ir/hlo_instructions.h"
#include "machina/xla/runtime/work_dimensions.h"
#include "machina/xla/service/buffer_assignment.h"
#include "machina/xla/shape.h"

namespace xla::emitters {

// Generic loop fusion.
class LoopFusionKernelEmitter final : public MlirKernelEmitter {
 public:
  LoopFusionKernelEmitter(mlir::MLIRContext& mlir_context,
                          const HloFusionInstruction& fusion,
                          const HloFusionSpec& fusion_spec,
                          const BufferAssignment* buffer_assignment,
                          KernelArguments::BufferAlignment buffer_alignment,
                          WorkDimensions work_dimensions,
                          absl::string_view entry_function_name,
                          BackendKind backend_kind);

  absl::StatusOr<MlirKernelDefinition> EmitKernelDefinition() override;

  static IndexingMap ComputeWorkItemIdToOutputIndexing(
      const WorkDimensions& work_dimensions, const Shape& root_shape,
      mlir::MLIRContext* ctx);

  // Get the shape that will be used for loop indexing for the given fusion
  // specification.
  static Shape GetIndexingShape(const HloFusionSpec& fusion_spec);

  std::string name() const final { return "loop_fusion_kernel_emitter"; }

 private:
  IndexingMap ComputeWorkItemIdToOutputIndexing(mlir::MLIRContext* ctx) const;
  absl::StatusOr<KernelSpec> GetKernelSpec() const;

  absl::Status EmitEntryFunction(
      const emitters::PartitionedComputations& computations,
      const emitters::CallTargetProvider& call_targets,
      mlir::func::FuncOp entry_function,
      const HloFusionInstruction& fusion) const;

 private:
  mlir::MLIRContext& mlir_context_;
  const HloFusionInstruction& fusion_;
  const HloFusionSpec& fusion_spec_;
  const BufferAssignment* buffer_assignment_;
  KernelArguments::BufferAlignment buffer_alignment_;
  WorkDimensions work_dimensions_;
  std::string entry_function_name_;
  BackendKind backend_kind_;
};

}  // namespace xla::emitters

#endif  // MACHINA_MACHINA_XLA_CODEGEN_EMITTERS_LOOP_KERNEL_EMITTER_H_
