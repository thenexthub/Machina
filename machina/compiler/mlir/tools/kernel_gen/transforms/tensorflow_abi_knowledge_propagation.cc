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

// This file contains the analysis and transformation to rewrite kernel
// functions such that information about alignment, aliasing and zero offsets
// steming from the tf_framework uses is propagated.

#include <cstdint>
#include <memory>

#include "toolchain/ADT/Bitfields.h"
#include "toolchain/ADT/DenseMap.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // part of Codira Toolchain
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/Dialect/GPU/IR/GPUDialect.h"  // part of Codira Toolchain
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // part of Codira Toolchain
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypes.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/tools/kernel_gen/ir/tf_framework_ops.h"
#include "machina/compiler/mlir/tools/kernel_gen/transforms/passes.h"

namespace mlir {
namespace kernel_gen {
namespace transforms {
namespace {

#define GEN_PASS_DEF_PROPAGATETFABIKNOWLEDGETOKERNELS
#include "machina/compiler/mlir/tools/kernel_gen/transforms/kernel_gen_passes.h.inc"

struct PropagateTfAbiKnowledgeToKernelsPass
    : public impl::PropagateTfAbiKnowledgeToKernelsBase<
          PropagateTfAbiKnowledgeToKernelsPass> {
  void runOnOperation() override {
    func::FuncOp function = getOperation();
    toolchain::SmallVector<Value, 4> worklist;
    // We currently only handle entry functions and do not propagate across
    // functions.
    if (function->getAttrOfType<mlir::UnitAttr>(
            tf_framework::TFFrameworkDialect::kTFEntryAttrName)) {
      // For all operands of this function, we know they are aligned. Also, by
      // construction of kernel generator, we know that there is no offset and
      // the inner stride is one.
      // TODO(herhut): Insert asserts in debug mode to check this.
      for (auto argument : function.getArguments()) {
        if (mlir::isa<BaseMemRefType>(argument.getType())) {
          worklist.push_back(argument);
          allocated_by_tf_runtime.insert(argument);
          offset_is_zero.insert(argument);
          inner_stride_is_constant.insert({argument, 1});
        }
      }
    }

    // For locally allocated values, we know they are aligned and have offset
    // zero. Further, they also do not alias with other memrefs, except in
    // benign ways. This is by construction and ensured by the reuse analysis.
    function.walk([&](tf_framework::TFAllocOp op) {
      Value allocated = op.getResult();
      worklist.push_back(allocated);
      no_alias.insert(allocated);
      allocated_by_tf_runtime.insert(allocated);
      offset_is_zero.insert(allocated);
      inner_stride_is_constant.insert({allocated, 1});
    });

    // Next, take what we have and propagate it through known operations.
    propagateThroughUses(worklist);

    // Now look at launches and make use of the knowledge we have.
    function.walk([&](gpu::LaunchFuncOp launch) {
      auto module = launch->getParentOfType<ModuleOp>();
      auto kernel = module.lookupSymbol<LLVM::LLVMFuncOp>(launch.getKernel());

      if (!kernel || kernel.isExternal()) return;

      // Count the position of kernel operands independently, as they do not
      // coincide with laucnh operands as memref parameters get expanded when
      // lowered to toolchain.
      int kernel_p = 0;
      OpBuilder b = OpBuilder::atBlockBegin(&kernel.getBody().front());
      toolchain::SmallDenseMap<int64_t, Value> constants;
      auto loc = kernel.getLoc();
      for (auto operand : launch.getKernelOperands()) {
        auto memref = mlir::dyn_cast<MemRefType>(operand.getType());
        if (!memref) {
          // Scalar argument, advance kernel position by one.
          kernel_p++;
          continue;
        }
        if (allocated_by_tf_runtime.contains(operand)) {
          // This was allocated by the tf runtime, so the two pointers in the
          // descriptor coincide. Rewrite the kernel accordingly.
          Value alloc_ptr = kernel.getArgument(kernel_p);
          Value align_ptr = kernel.getArgument(kernel_p + 1);
          alloc_ptr.replaceAllUsesWith(align_ptr);
          kernel.setArgAttr(
              kernel_p + 1, LLVM::LLVMDialect::getAlignAttrName(),
              b.getIndexAttr(
                  tf_framework::TFFrameworkDialect::kAllocationAlignment));
        }
        if (offset_is_zero.contains(operand)) {
          Value offset = kernel.getArgument(kernel_p + 2);
          Value &zero = constants[0];
          if (!zero) {
            zero = b.create<LLVM::ConstantOp>(loc, offset.getType(),
                                              b.getIndexAttr(0));
          }
          offset.replaceAllUsesWith(zero);
        }
        auto const_stride = inner_stride_is_constant.find(operand);
        if (const_stride != inner_stride_is_constant.end()) {
          // The stride is the last argument belonging to this memref.
          Value inner_stride =
              kernel.getArgument(kernel_p + 2 + memref.getRank() * 2);
          Value &stride_val = constants[const_stride->second];
          if (!stride_val) {
            stride_val = b.create<LLVM::ConstantOp>(
                loc, inner_stride.getType(),
                b.getIndexAttr(const_stride->second));
          }
          inner_stride.replaceAllUsesWith(stride_val);
        }
        if (no_alias.contains(operand)) {
          // TODO(herhut): We also need to check whether any of the other args
          //     are aliases. This is currently never the case by construction
          //     but we could use the alias analysis from buffer placement here
          //     to make sure.
          // Add the no_alias attribute to the corresponding pointer.
          kernel.setArgAttr(kernel_p + 1,
                            LLVM::LLVMDialect::getNoAliasAttrName(),
                            b.getUnitAttr());
        }
        // Advance base, aligned, offset, strides and sizes many arguments.
        kernel_p += memref.getRank() * 2 + 3;
      }
    });
  }

 private:
  void propagateThroughUses(SmallVectorImpl<Value> &worklist) {
    while (!worklist.empty()) {
      Value candidate = worklist.pop_back_val();
      for (auto user : candidate.getUsers()) {
        if (isa<memref::CastOp, memref::ReshapeOp>(user)) {
          // Reshape and Cast propagate alignment, offset and innermost stride.
          // TODO(herhut): This should be a trait.
          Value result = user->getResult(0);
          if (allocated_by_tf_runtime.contains(candidate)) {
            allocated_by_tf_runtime.insert(result);
          }
          auto const_stride = inner_stride_is_constant.find(candidate);
          if (const_stride != inner_stride_is_constant.end()) {
            inner_stride_is_constant.insert({result, const_stride->second});
          }
          if (offset_is_zero.contains(candidate)) {
            offset_is_zero.insert(result);
          }
          worklist.push_back(result);
        }
        if (auto cast = dyn_cast<memref::ReinterpretCastOp>(user)) {
          // Check that we have offset 0.
          Value result = cast.getResult();
          if (!cast.isDynamicOffset(0) && cast.getStaticOffset(0) == 0) {
            offset_is_zero.insert(result);
          }
          if (allocated_by_tf_runtime.contains(candidate)) {
            allocated_by_tf_runtime.insert(result);
          }
          size_t last_stride = cast.getResultRank() - 1;
          // TODO(herhut): Remove this once canonicalization handles this.
          if (cast.isDynamicStride(last_stride)) {
            auto dyn_stride = cast.getDynamicStride(last_stride)
                                  .getDefiningOp<arith::ConstantIndexOp>();
            if (dyn_stride) {
              inner_stride_is_constant.insert({result, dyn_stride.value()});
            }
          } else {
            inner_stride_is_constant.insert(
                {result, cast.getStaticStride(last_stride)});
          }
          worklist.push_back(result);
        }
      }
    }
  }

  // Set of values that were allocated by the tf runtime and hence are aligned.
  toolchain::SmallPtrSet<Value, 8> allocated_by_tf_runtime;
  // Set of values that are known to not have an offset of 0.
  toolchain::SmallPtrSet<Value, 8> offset_is_zero;
  // Set of values that are known to have a constant stride.
  toolchain::SmallDenseMap<Value, int64_t, 8> inner_stride_is_constant;
  // Set of values we know do not alias other values.
  toolchain::SmallPtrSet<Value, 8> no_alias;
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
CreatePropagateTfAbiKnowledgeToKernels() {
  return std::make_unique<PropagateTfAbiKnowledgeToKernelsPass>();
}

}  // namespace transforms
}  // namespace kernel_gen
}  // namespace mlir
