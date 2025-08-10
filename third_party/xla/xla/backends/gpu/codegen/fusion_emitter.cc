/* Copyright 2023 The OpenXLA Authors.

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
#include "machina/xla/backends/gpu/codegen/fusion_emitter.h"

#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "toolchain/ADT/STLExtras.h"
#include "toolchain/ADT/SmallVector.h"
#include "toolchain/IR/Argument.h"
#include "toolchain/IR/Attributes.h"
#include "toolchain/IR/BasicBlock.h"
#include "toolchain/IR/DerivedTypes.h"
#include "toolchain/IR/Function.h"
#include "toolchain/IR/GlobalValue.h"
#include "toolchain/IR/IRBuilder.h"
#include "toolchain/IR/Instructions.h"
#include "toolchain/IR/Metadata.h"
#include "toolchain/IR/Type.h"
#include "toolchain/TargetParser/Triple.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "machina/xla/codegen/emitters/kernel_api_builder.h"
#include "machina/xla/codegen/emitters/kernel_arguments.h"
#include "machina/xla/hlo/analysis/indexing_map.h"
#include "machina/xla/runtime/work_dimensions.h"
#include "machina/xla/service/gpu/ir_emitter_context.h"
#include "machina/xla/service/gpu/launch_dimensions.h"
#include "machina/xla/service/gpu/target_util.h"
#include "machina/xla/service/llvm_ir/llvm_util.h"
#include "machina/xla/shape.h"
#include "machina/xla/status_macros.h"
#include "machina/xla/stream_executor/device_description.h"
#include "machina/xla/tsl/platform/errors.h"

namespace xla {
namespace gpu {

// Annotates the launch dimensions of the corresponding IR kernel in
// `llvm_module`.
absl::Status AnnotateKernelLaunchDimensions(
    const se::DeviceDescription& device_info,
    const LaunchDimensions& launch_dims, toolchain::Function* kernel,
    toolchain::Module* llvm_module) {
  TF_RET_CHECK(
      (device_info.block_dim_limit().x == 0 ||
       launch_dims.block_counts().x < device_info.block_dim_limit().x) &&
      (device_info.block_dim_limit().y == 0 ||
       launch_dims.block_counts().y < device_info.block_dim_limit().y))
      << "Kernel '" << kernel->getName().str() << "' launch needs more blocks ("
      << launch_dims.block_counts().x << ", " << launch_dims.block_counts().y
      << ") than allowed by hardware (" << device_info.block_dim_limit().x
      << ", " << device_info.block_dim_limit().y << ").";
  // Add __launch_bounds__ to metadata. This limits registers per thread to
  // avoid out-of-resources launching errors.

  toolchain::Triple target_triple = toolchain::Triple(llvm_module->getTargetTriple());

  if (target_triple.isNVPTX()) {
    // Our launch bounds are exact, so we can specify them as
    // reqntid[xyz] rather than maxntid[xyz].
    const std::string attr =
        absl::StrCat(launch_dims.thread_counts_per_block().x, ",",
                     launch_dims.thread_counts_per_block().y, ",",
                     launch_dims.thread_counts_per_block().z);
    kernel->addFnAttr("nvvm.reqntid", attr);
    // Maybe we want to set "reqnctapercluster" here, but not sure if needed or
    // if LLVM supports that yet. Let's do that later when needed.
  } else if (target_triple.getArch() == toolchain::Triple::amdgcn) {
    kernel->addFnAttr("amdgpu-flat-work-group-size",
                      absl::StrJoin({launch_dims.num_threads_per_block(),
                                     launch_dims.num_threads_per_block()},
                                    ","));
    kernel->addFnAttr("amdgpu-max-num-workgroups",
                      absl::StrJoin({launch_dims.block_counts().x,
                                     launch_dims.block_counts().y,
                                     launch_dims.block_counts().z},
                                    ","));
  }
  return absl::OkStatus();
}

IndexingMap KernelFusionInterface::GetDefaultThreadIdIndexingMap(
    const LaunchDimensions& launch_dims, int unroll_factor, const Shape& shape,
    mlir::MLIRContext* ctx) {
  WorkDimensions work_dimensions = launch_dims.AsWorkDimensions();
  work_dimensions.work_tile_size.dimensions.push_back(unroll_factor);
  return emitters::GetDefaultWorkItemIndexingMap(work_dimensions, shape, ctx);
}

std::string GetSanitizedUniqueName(IrEmitterContext& ir_emitter_context,
                                   const std::string& suggested_name) {
  return ir_emitter_context.name_uniquer()->GetUniqueName(
      llvm_ir::SanitizeFunctionName(suggested_name));
}

absl::StatusOr<toolchain::Function*> BuildKernelPrototype(
    IrEmitterContext& ir_emitter_context, const std::string& impl_fn_name,
    const std::string& suggested_name,
    const emitters::KernelArguments& arguments,
    const LaunchDimensions& launch_dimensions, toolchain::IRBuilderBase* builder) {
  return BuildKernelPrototypeFromUniqueName(
      ir_emitter_context, impl_fn_name,
      GetSanitizedUniqueName(ir_emitter_context, suggested_name), arguments,
      launch_dimensions, builder);
}

absl::StatusOr<toolchain::Function*> BuildKernelPrototypeFromUniqueName(
    IrEmitterContext& ir_emitter_context, const std::string& impl_fn_name,
    const std::string& unique_kernel_name,
    const emitters::KernelArguments& arguments,
    const LaunchDimensions& launch_dimensions, toolchain::IRBuilderBase* builder) {
  // Create the kernel and add it to the module.
  auto* llvm_module = ir_emitter_context.llvm_module();
  toolchain::LLVMContext& context = llvm_module->getContext();
  // Explicitly set global addrspace for SPIR backend.
  int addrspace = toolchain::Triple(llvm_module->getTargetTriple()).isSPIR() ? 1 : 0;
  toolchain::FunctionType* kernel_type = toolchain::FunctionType::get(
      /*Result=*/toolchain::Type::getVoidTy(context),
      std::vector<toolchain::Type*>(arguments.args().size(),
                               builder->getPtrTy(addrspace)),
      /*isVarArg=*/false);
  toolchain::Function* kernel =
      toolchain::Function::Create(kernel_type, toolchain::GlobalValue::ExternalLinkage,
                             unique_kernel_name, llvm_module);

  AnnotateFunctionAsGpuKernel(llvm_module, kernel, builder);
  TF_RETURN_IF_ERROR(
      AnnotateKernelLaunchDimensions(ir_emitter_context.gpu_device_info(),
                                     launch_dimensions, kernel, llvm_module));

  // Update the insert point to the entry basic block.
  toolchain::BasicBlock* entry_bb =
      toolchain::BasicBlock::Create(context, /*Name=*/"entry", /*Parent=*/kernel);

  // Emit a "return void" at entry_bb's end, and set the insert point before
  // that return instruction.
  builder->SetInsertPoint(toolchain::ReturnInst::Create(context, entry_bb));
  // Get the original function to extract attributes.
  auto impl_func = llvm_module->getFunction(impl_fn_name);

  for (auto&& [arg_idx, kernel_argument] : toolchain::enumerate(arguments.args())) {
    // Get the original argument to extract attributes from if they exist.
    toolchain::Argument* impl_arg = impl_func ? impl_func->getArg(arg_idx) : nullptr;
    toolchain::Argument& llvm_arg = *kernel->getArg(arg_idx);
    llvm_arg.setName(absl::StrCat("arg", arg_idx));

    if (impl_arg && impl_arg->hasByValAttr()) {
      kernel->addParamAttr(arg_idx,
                           impl_arg->getAttribute(toolchain::Attribute::ByVal));
    } else {
      kernel->addDereferenceableParamAttr(arg_idx,
                                          kernel_argument.slice().size());
    }
    // If the alignment has been specified in the original function, use it.
    // Otherwise, use the alignment from the kernel argument.
    if (impl_arg && impl_arg->hasAttribute(toolchain::Attribute::Alignment)) {
      kernel->addParamAttr(arg_idx,
                           impl_arg->getAttribute(toolchain::Attribute::Alignment));
    } else {
      kernel->addParamAttr(arg_idx,
                           toolchain::Attribute::get(llvm_arg.getContext(),
                                                toolchain::Attribute::Alignment,
                                                kernel_argument.alignment()));
    }
    if (!kernel_argument.aliased()) {
      kernel->addParamAttr(arg_idx,
                           toolchain::Attribute::get(llvm_arg.getContext(),
                                                toolchain::Attribute::NoAlias));
    }
  }

  return kernel;
}

}  // namespace gpu
}  // namespace xla
