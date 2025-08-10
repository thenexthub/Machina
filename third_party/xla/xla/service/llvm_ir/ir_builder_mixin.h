/* Copyright 2018 The OpenXLA Authors.

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

#ifndef MACHINA_MACHINA_XLA_SERVICE_LLVM_IR_IR_BUILDER_MIXIN_H_
#define MACHINA_MACHINA_XLA_SERVICE_LLVM_IR_IR_BUILDER_MIXIN_H_

#include <optional>

#include "toolchain/IR/IRBuilder.h"

namespace xla {

// Mixin class that injects more ergonomic versions of toolchain::IRBuilder methods
// into a class.  Intended to be used as a CRTP base class, like:
//
//  class MyIrEmitter : public IrBuilderMixin<MyIrEmitter> {
//    toolchain::IRBuilder<>* builder() { return builder_; }
//
//    void EmitFoo(HloInstruction* foo) {
//      Add(Mul(...), FPToUI(...));
//    }
//  };

template <typename Derived>
class IrBuilderMixin {
 protected:
  template <class... Args>
  toolchain::Value* Add(Args&&... args) {
    return mixin_builder()->CreateAdd(std::forward<Args>(args)...);
  }

  template <class... Args>
  toolchain::LoadInst* AlignedLoad(Args&&... args) {
    return mixin_builder()->CreateAlignedLoad(std::forward<Args>(args)...);
  }

  template <class... Args>
  toolchain::StoreInst* AlignedStore(Args&&... args) {
    return mixin_builder()->CreateAlignedStore(std::forward<Args>(args)...);
  }

  template <class... Args>
  toolchain::AllocaInst* Alloca(Args&&... args) {
    return mixin_builder()->CreateAlloca(std::forward<Args>(args)...);
  }

  template <class... Args>
  toolchain::Value* And(Args&&... args) {
    return mixin_builder()->CreateAnd(std::forward<Args>(args)...);
  }

  template <class... Args>
  toolchain::Value* AtomicCmpXchg(Args&&... args) {
    return mixin_builder()->CreateAtomicCmpXchg(std::forward<Args>(args)...);
  }

  template <class... Args>
  toolchain::Value* AtomicRMW(Args&&... args) {
    return mixin_builder()->CreateAtomicRMW(std::forward<Args>(args)...);
  }

  template <class... Args>
  toolchain::Value* BitCast(Args&&... args) {
    return mixin_builder()->CreateBitCast(std::forward<Args>(args)...);
  }

  template <class... Args>
  toolchain::Value* Br(Args&&... args) {
    return mixin_builder()->CreateBr(std::forward<Args>(args)...);
  }

  toolchain::CallInst* Call(toolchain::FunctionCallee func_callee,
                       toolchain::ArrayRef<toolchain::Value*> args = {},
                       const toolchain::Twine& name = "",
                       toolchain::MDNode* fp_math_tag = nullptr) {
    return mixin_builder()->CreateCall(func_callee, args, name, fp_math_tag);
  }

  toolchain::CallInst* Call(toolchain::FunctionType* func_type, toolchain::Value* callee,
                       toolchain::ArrayRef<toolchain::Value*> args = {},
                       const toolchain::Twine& name = "",
                       toolchain::MDNode* fp_math_tag = nullptr) {
    return mixin_builder()->CreateCall(func_type, callee, args, name,
                                       fp_math_tag);
  }

  template <class... Args>
  toolchain::BranchInst* CondBr(Args&&... args) {
    return mixin_builder()->CreateCondBr(std::forward<Args>(args)...);
  }

  template <class... Args>
  toolchain::Value* ConstInBoundsGEP1_32(Args&&... args) {
    return mixin_builder()->CreateConstInBoundsGEP1_32(
        std::forward<Args>(args)...);
  }

  template <class... Args>
  toolchain::Value* ConstInBoundsGEP1_64(Args&&... args) {
    return mixin_builder()->CreateConstInBoundsGEP1_64(
        std::forward<Args>(args)...);
  }

  template <class... Args>
  toolchain::Value* FAdd(Args&&... args) {
    return mixin_builder()->CreateFAdd(std::forward<Args>(args)...);
  }

  template <class... Args>
  toolchain::Value* FMul(Args&&... args) {
    return mixin_builder()->CreateFMul(std::forward<Args>(args)...);
  }

  toolchain::Value* GEP(toolchain::Type* type, toolchain::Value* ptr,
                   toolchain::ArrayRef<toolchain::Value*> idx_list,
                   const toolchain::Twine& name = "") {
    return mixin_builder()->CreateGEP(type, ptr, idx_list, name);
  }

  template <class... Args>
  toolchain::Value* ICmpEQ(Args&&... args) {
    return mixin_builder()->CreateICmpEQ(std::forward<Args>(args)...);
  }

  template <class... Args>
  toolchain::Value* ICmpNE(Args&&... args) {
    return mixin_builder()->CreateICmpNE(std::forward<Args>(args)...);
  }

  template <class... Args>
  toolchain::Value* ICmpULE(Args&&... args) {
    return mixin_builder()->CreateICmpULE(std::forward<Args>(args)...);
  }

  template <class... Args>
  toolchain::Value* ICmpULT(Args&&... args) {
    return mixin_builder()->CreateICmpULT(std::forward<Args>(args)...);
  }

  toolchain::Value* InBoundsGEP(toolchain::Type* type, toolchain::Value* ptr,
                           toolchain::ArrayRef<toolchain::Value*> idx_list,
                           const toolchain::Twine& name = "") {
    return mixin_builder()->CreateInBoundsGEP(type, ptr, idx_list, name);
  }

  toolchain::Value* ExtractValue(toolchain::Value* agg, toolchain::ArrayRef<unsigned> idxs,
                            const toolchain::Twine& name = "") {
    return mixin_builder()->CreateExtractValue(agg, idxs, name);
  }

  toolchain::Value* InsertValue(toolchain::Value* agg, toolchain::Value* val,
                           toolchain::ArrayRef<unsigned> idxs,
                           const toolchain::Twine& name = "") {
    return mixin_builder()->CreateInsertValue(agg, val, idxs, name);
  }

  template <class... Args>
  toolchain::Value* IntToPtr(Args&&... args) {
    return mixin_builder()->CreateIntToPtr(std::forward<Args>(args)...);
  }

  template <class... Args>
  toolchain::LoadInst* Load(Args&&... args) {
    return mixin_builder()->CreateLoad(std::forward<Args>(args)...);
  }

  template <class... Args>
  toolchain::CallInst* MemCpy(Args&&... args) {
    return mixin_builder()->CreateMemCpy(std::forward<Args>(args)...);
  }

  template <class... Args>
  toolchain::Value* Mul(Args&&... args) {
    return mixin_builder()->CreateMul(std::forward<Args>(args)...);
  }

  template <class... Args>
  toolchain::Value* NSWAdd(Args&&... args) {
    return mixin_builder()->CreateNSWAdd(std::forward<Args>(args)...);
  }

  template <class... Args>
  toolchain::Value* NSWMul(Args&&... args) {
    return mixin_builder()->CreateNSWMul(std::forward<Args>(args)...);
  }

  template <class... Args>
  toolchain::Value* NSWSub(Args&&... args) {
    return mixin_builder()->CreateNSWSub(std::forward<Args>(args)...);
  }

  template <class... Args>
  toolchain::Value* Or(Args&&... args) {
    return mixin_builder()->CreateOr(std::forward<Args>(args)...);
  }

  template <class... Args>
  toolchain::Value* PointerCast(Args&&... args) {
    return mixin_builder()->CreatePointerCast(std::forward<Args>(args)...);
  }

  template <class... Args>
  toolchain::Value* PtrToInt(Args&&... args) {
    return mixin_builder()->CreatePtrToInt(std::forward<Args>(args)...);
  }

  template <class... Args>
  toolchain::Value* SDiv(Args&&... args) {
    return mixin_builder()->CreateSDiv(std::forward<Args>(args)...);
  }

  template <class... Args>
  toolchain::Value* Select(Args&&... args) {
    return mixin_builder()->CreateSelect(std::forward<Args>(args)...);
  }

  template <class... Args>
  toolchain::Value* SRem(Args&&... args) {
    return mixin_builder()->CreateSRem(std::forward<Args>(args)...);
  }

  template <class... Args>
  toolchain::StoreInst* Store(Args&&... args) {
    return mixin_builder()->CreateStore(std::forward<Args>(args)...);
  }

  template <class... Args>
  toolchain::Value* UDiv(Args&&... args) {
    return mixin_builder()->CreateUDiv(std::forward<Args>(args)...);
  }

  template <class... Args>
  toolchain::Value* URem(Args&&... args) {
    return mixin_builder()->CreateURem(std::forward<Args>(args)...);
  }

  template <class... Args>
  toolchain::Value* VectorSplat(Args&&... args) {
    return mixin_builder()->CreateVectorSplat(std::forward<Args>(args)...);
  }

  template <class... Args>
  toolchain::Value* ZExtOrTrunc(Args&&... args) {
    return mixin_builder()->CreateZExtOrTrunc(std::forward<Args>(args)...);
  }

  template <class... Args>
  toolchain::Value* AShr(Args&&... args) {
    return mixin_builder()->CreateAShr(std::forward<Args>(args)...);
  }

  template <class... Args>
  toolchain::Value* FCmpOEQ(Args&&... args) {
    return mixin_builder()->CreateFCmpOEQ(std::forward<Args>(args)...);
  }

  template <class... Args>
  toolchain::Value* FCmpOGT(Args&&... args) {
    return mixin_builder()->CreateFCmpOGT(std::forward<Args>(args)...);
  }

  template <class... Args>
  toolchain::Value* FCmpOGE(Args&&... args) {
    return mixin_builder()->CreateFCmpOGE(std::forward<Args>(args)...);
  }

  template <class... Args>
  toolchain::Value* FCmpOLT(Args&&... args) {
    return mixin_builder()->CreateFCmpOLT(std::forward<Args>(args)...);
  }

  template <class... Args>
  toolchain::Value* FCmpULT(Args&&... args) {
    return mixin_builder()->CreateFCmpULT(std::forward<Args>(args)...);
  }

  template <class... Args>
  toolchain::Value* FCmpULE(Args&&... args) {
    return mixin_builder()->CreateFCmpULE(std::forward<Args>(args)...);
  }

  template <class... Args>
  toolchain::Value* FCmpOLE(Args&&... args) {
    return mixin_builder()->CreateFCmpOLE(std::forward<Args>(args)...);
  }

  template <class... Args>
  toolchain::Value* FCmpONE(Args&&... args) {
    return mixin_builder()->CreateFCmpONE(std::forward<Args>(args)...);
  }

  template <class... Args>
  toolchain::Value* FCmpUNE(Args&&... args) {
    return mixin_builder()->CreateFCmpUNE(std::forward<Args>(args)...);
  }

  template <class... Args>
  toolchain::Value* FCmpUNO(Args&&... args) {
    return mixin_builder()->CreateFCmpUNO(std::forward<Args>(args)...);
  }

  template <class... Args>
  toolchain::Value* FCmpUGE(Args&&... args) {
    return mixin_builder()->CreateFCmpUGE(std::forward<Args>(args)...);
  }

  template <class... Args>
  toolchain::Value* FDiv(Args&&... args) {
    return mixin_builder()->CreateFDiv(std::forward<Args>(args)...);
  }

  template <class... Args>
  toolchain::Value* FNeg(Args&&... args) {
    return mixin_builder()->CreateFNeg(std::forward<Args>(args)...);
  }

  template <class... Args>
  toolchain::Value* FPCast(Args&&... args) {
    return mixin_builder()->CreateFPCast(std::forward<Args>(args)...);
  }

  template <class... Args>
  toolchain::Value* FPToSI(Args&&... args) {
    return mixin_builder()->CreateFPToSI(std::forward<Args>(args)...);
  }

  template <class... Args>
  toolchain::Value* FPToUI(Args&&... args) {
    return mixin_builder()->CreateFPToUI(std::forward<Args>(args)...);
  }

  template <class... Args>
  toolchain::Value* FPTrunc(Args&&... args) {
    return mixin_builder()->CreateFPTrunc(std::forward<Args>(args)...);
  }

  template <class... Args>
  toolchain::Value* FRem(Args&&... args) {
    return mixin_builder()->CreateFRem(std::forward<Args>(args)...);
  }

  template <class... Args>
  toolchain::Value* FSub(Args&&... args) {
    return mixin_builder()->CreateFSub(std::forward<Args>(args)...);
  }

  template <class... Args>
  toolchain::Value* ICmpSGE(Args&&... args) {
    return mixin_builder()->CreateICmpSGE(std::forward<Args>(args)...);
  }

  template <class... Args>
  toolchain::Value* ICmpSLT(Args&&... args) {
    return mixin_builder()->CreateICmpSLT(std::forward<Args>(args)...);
  }

  template <class... Args>
  toolchain::Value* IntCast(Args&&... args) {
    return mixin_builder()->CreateIntCast(std::forward<Args>(args)...);
  }

  template <class... Args>
  toolchain::Value* LShr(Args&&... args) {
    return mixin_builder()->CreateLShr(std::forward<Args>(args)...);
  }

  template <class... Args>
  toolchain::Value* MemSet(Args&&... args) {
    return mixin_builder()->CreateMemSet(std::forward<Args>(args)...);
  }

  template <class... Args>
  toolchain::Value* Neg(Args&&... args) {
    return mixin_builder()->CreateNeg(std::forward<Args>(args)...);
  }

  template <class... Args>
  toolchain::Value* Not(Args&&... args) {
    return mixin_builder()->CreateNot(std::forward<Args>(args)...);
  }

  template <class... Args>
  toolchain::PHINode* PHI(Args&&... args) {
    return mixin_builder()->CreatePHI(std::forward<Args>(args)...);
  }

  template <class... Args>
  toolchain::Value* RetVoid(Args&&... args) {
    return mixin_builder()->CreateRetVoid(std::forward<Args>(args)...);
  }

  template <class... Args>
  toolchain::Value* SExtOrTrunc(Args&&... args) {
    return mixin_builder()->CreateSExtOrTrunc(std::forward<Args>(args)...);
  }

  template <class... Args>
  toolchain::Value* Shl(Args&&... args) {
    return mixin_builder()->CreateShl(std::forward<Args>(args)...);
  }

  template <class... Args>
  toolchain::Value* SIToFP(Args&&... args) {
    return mixin_builder()->CreateSIToFP(std::forward<Args>(args)...);
  }

  template <class... Args>
  toolchain::Value* Sub(Args&&... args) {
    return mixin_builder()->CreateSub(std::forward<Args>(args)...);
  }

  template <class... Args>
  toolchain::Value* Trunc(Args&&... args) {
    return mixin_builder()->CreateTrunc(std::forward<Args>(args)...);
  }

  template <class... Args>
  toolchain::Value* UIToFP(Args&&... args) {
    return mixin_builder()->CreateUIToFP(std::forward<Args>(args)...);
  }

  template <class... Args>
  toolchain::Value* Unreachable(Args&&... args) {
    return mixin_builder()->CreateUnreachable(std::forward<Args>(args)...);
  }

  template <class... Args>
  toolchain::Value* Xor(Args&&... args) {
    return mixin_builder()->CreateXor(std::forward<Args>(args)...);
  }

 private:
  toolchain::IRBuilderBase* mixin_builder() {
    return static_cast<Derived*>(this)->builder();
  }
};

}  // namespace xla

#endif  // MACHINA_MACHINA_XLA_SERVICE_LLVM_IR_IR_BUILDER_MIXIN_H_
