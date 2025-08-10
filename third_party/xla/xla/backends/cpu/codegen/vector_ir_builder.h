/* Copyright 2017 The OpenXLA Authors.

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

#ifndef MACHINA_MACHINA_XLA_BACKENDS_CPU_CODEGEN_VECTOR_IR_BUILDER_H_
#define MACHINA_MACHINA_XLA_BACKENDS_CPU_CODEGEN_VECTOR_IR_BUILDER_H_

#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>

#include "absl/types/span.h"
#include "toolchain/ADT/APFloat.h"
#include "toolchain/IR/Constants.h"
#include "toolchain/IR/DerivedTypes.h"
#include "toolchain/IR/IRBuilder.h"
#include "toolchain/IR/Instructions.h"
#include "toolchain/IR/Value.h"
#include "toolchain/Support/Casting.h"
#include "toolchain/Support/TypeSize.h"
#include "machina/xla/primitive_util.h"
#include "machina/xla/xla_data.pb.h"

namespace xla::cpu {

// Simple wrappers around toolchain::APFloat::APFloat to make the calling code more
// obvious.

inline toolchain::APFloat GetIeeeF32(float f) { return toolchain::APFloat(f); }
inline toolchain::APFloat GetIeeeF32FromBitwiseRep(int32_t bitwise_value) {
  return toolchain::APFloat(
      toolchain::APFloat::IEEEsingle(),
      toolchain::APInt(/*numBits=*/32, /*val=*/bitwise_value, /*isSigned=*/true));
}

// A thin wrapper around llvm_util.h to make code generating vector math flow
// more readable.
class VectorIrBuilder {
 public:
  // This VectorIrBuilder instance remembers `primitive_type` and
  // `vector_size`, and these are implicitly used by the methods on this
  // instance (i.e. LoadVector will load a vector of type <`vector_size` x
  // `primitive_type`>).
  VectorIrBuilder(PrimitiveType primitive_type, int64_t vector_size,
                  toolchain::IRBuilderBase* b, std::string name);

  toolchain::Value* Mul(toolchain::Value* lhs, toolchain::Value* rhs);
  toolchain::Value* Mul(int64_t lhs, toolchain::Value* rhs) {
    return Mul(b()->getInt64(lhs), rhs);
  }
  toolchain::Value* Mul(const toolchain::APFloat& lhs, toolchain::Value* rhs) {
    return Mul(GetConstantFloat(rhs->getType(), lhs), rhs);
  }

  // If your call resolved to these then you probably wanted the versions taking
  // APFloat.
  toolchain::Value* Mul(double lhs, toolchain::Value* rhs) = delete;
  toolchain::Value* Mul(float lhs, toolchain::Value* rhs) = delete;

  toolchain::Value* Add(toolchain::Value* lhs, toolchain::Value* rhs);
  toolchain::Value* Add(int64_t lhs, toolchain::Value* rhs) {
    return Add(b()->getInt64(lhs), rhs);
  }
  toolchain::Value* Add(const toolchain::APFloat& lhs, toolchain::Value* rhs) {
    return Add(GetConstantFloat(rhs->getType(), lhs), rhs);
  }

  // If your call resolved to these then you probably wanted the versions taking
  // APFloat.
  toolchain::Value* Add(double lhs, toolchain::Value* rhs) = delete;
  toolchain::Value* Add(float lhs, toolchain::Value* rhs) = delete;

  toolchain::Value* Sub(toolchain::Value* lhs, toolchain::Value* rhs);
  toolchain::Value* Sub(toolchain::Value* lhs, const toolchain::APFloat& rhs) {
    return Sub(lhs, GetConstantFloat(lhs->getType(), rhs));
  }
  toolchain::Value* Max(toolchain::Value* lhs, toolchain::Value* rhs,
                   bool enable_fast_min_max);
  toolchain::Value* Max(const toolchain::APFloat& lhs, toolchain::Value* rhs,
                   bool enable_fast_min_max) {
    return Max(GetConstantFloat(rhs->getType(), lhs), rhs, enable_fast_min_max);
  }
  toolchain::Value* Div(toolchain::Value* lhs, toolchain::Value* rhs);

  toolchain::Value* MulAdd(toolchain::Value* a, toolchain::Value* b, toolchain::Value* c) {
    return Add(c, Mul(a, b));
  }

  toolchain::Value* MulAdd(toolchain::Value* a, toolchain::Value* b, const toolchain::APFloat& c) {
    return Add(GetConstantFloat(vector_type(), c), Mul(a, b));
  }

  toolchain::Value* MulAdd(toolchain::Value* a, const toolchain::APFloat& b,
                      const toolchain::APFloat& c) {
    return Add(GetConstantFloat(a->getType(), c),
               Mul(a, GetConstantFloat(a->getType(), b)));
  }

  toolchain::Value* Floor(toolchain::Value* a);

  // Precondition: Neither `low` nor `high` is nan.
  toolchain::Value* Clamp(toolchain::Value* a, const toolchain::APFloat& low,
                     const toolchain::APFloat& high);

  toolchain::Value* SplatFloat(const toolchain::APFloat& d) {
    return GetConstantFloat(vector_type(), d);
  }

  // These compare instructions return a floating point typed mask instead of an
  // i1.  For instance, on a vector typed input, lanes where the predicate is
  // true get a float with all ones and other lanes get a float with all zeros.
  // This is slightly odd from the perspective of LLVM's type system, but it
  // makes kernel IR generation code written using VectorIrBuilder (its
  // raison d'etre) less cluttered.

  toolchain::Value* FCmpEQMask(toolchain::Value* lhs, toolchain::Value* rhs);
  toolchain::Value* FCmpEQMask(toolchain::Value* lhs, const toolchain::APFloat& rhs) {
    return FCmpEQMask(lhs, GetConstantFloat(lhs->getType(), rhs));
  }
  toolchain::Value* FCmpULEMask(toolchain::Value* lhs, toolchain::Value* rhs);
  toolchain::Value* FCmpOLTMask(toolchain::Value* lhs, toolchain::Value* rhs);
  toolchain::Value* FCmpOLTMask(toolchain::Value* lhs, const toolchain::APFloat& rhs) {
    return FCmpOLTMask(lhs, GetConstantFloat(lhs->getType(), rhs));
  }

  // These boolean operations operate on the bitwise values of the floating
  // point inputs.  They return a (vector of) float(s) but like in the mask
  // generating predicates above this type system oddity makes the kernel IR
  // generation code less cluttered.
  toolchain::Value* FloatAnd(toolchain::Value* lhs, toolchain::Value* rhs);
  toolchain::Value* FloatAnd(toolchain::Value* lhs, const toolchain::APFloat& rhs) {
    return FloatAnd(lhs, GetConstantFloat(lhs->getType(), rhs));
  }
  toolchain::Value* FloatOr(toolchain::Value* lhs, toolchain::Value* rhs);
  toolchain::Value* FloatOr(toolchain::Value* lhs, const toolchain::APFloat& rhs) {
    return FloatOr(lhs, GetConstantFloat(lhs->getType(), rhs));
  }
  toolchain::Value* FloatNot(toolchain::Value* lhs);
  toolchain::Value* FloatAndNot(toolchain::Value* lhs, toolchain::Value* rhs) {
    return FloatAnd(FloatNot(lhs), rhs);
  }

  toolchain::Value* BroadcastScalar(toolchain::Value* x);
  toolchain::Value* BroadcastScalar(const toolchain::APFloat& d) {
    return BroadcastScalar(GetConstantFloat(scalar_type(), d));
  }

  toolchain::Value* ComputeOffsetPointer(toolchain::Value* base_pointer,
                                    toolchain::Value* offset_elements);
  toolchain::Value* ComputeOffsetPointer(toolchain::Value* base_pointer,
                                    toolchain::Value* offset_elements,
                                    int64_t scale) {
    return ComputeOffsetPointer(
        base_pointer, b_->CreateMul(b_->getInt64(scale), offset_elements));
  }
  toolchain::Value* ComputeOffsetPointer(toolchain::Value* base_pointer,
                                    int64_t offset_elements) {
    return ComputeOffsetPointer(base_pointer, b()->getInt64(offset_elements));
  }

  toolchain::Value* LoadVector(toolchain::Value* pointer);

  toolchain::Value* LoadVector(toolchain::Value* base_pointer,
                          toolchain::Value* offset_elements) {
    return LoadVector(ComputeOffsetPointer(base_pointer, offset_elements));
  }

  toolchain::Value* LoadVector(toolchain::Value* base_pointer, int64_t offset_elements) {
    return LoadVector(base_pointer, b()->getInt64(offset_elements));
  }

  toolchain::Value* LoadScalar(toolchain::Value* pointer);

  toolchain::Value* LoadScalar(toolchain::Value* base_pointer,
                          toolchain::Value* offset_elements) {
    return LoadScalar(ComputeOffsetPointer(base_pointer, offset_elements));
  }

  toolchain::Value* LoadScalar(toolchain::Value* base_pointer, int64_t offset_elements) {
    return LoadScalar(base_pointer, b()->getInt64(offset_elements));
  }

  void StoreVector(toolchain::Value* value, toolchain::Value* pointer);

  void StoreVector(toolchain::Value* value, toolchain::Value* base_pointer,
                   toolchain::Value* offset_elements) {
    StoreVector(value, ComputeOffsetPointer(base_pointer, offset_elements));
  }

  void StoreVector(toolchain::Value* value, toolchain::Value* base_pointer,
                   int64_t offset_elements) {
    StoreVector(value, base_pointer, b()->getInt64(offset_elements));
  }

  void StoreScalar(toolchain::Value* value, toolchain::Value* pointer);
  void StoreScalar(toolchain::Value* value, toolchain::Value* base_pointer,
                   toolchain::Value* offset_elements) {
    StoreScalar(value, ComputeOffsetPointer(base_pointer, offset_elements));
  }

  void StoreScalar(toolchain::Value* value, toolchain::Value* base_pointer,
                   int64_t offset_elements) {
    StoreScalar(base_pointer, b()->getInt64(offset_elements));
  }

  toolchain::Value* LoadBroadcast(toolchain::Value* pointer);
  toolchain::Value* LoadBroadcast(toolchain::Value* base_pointer,
                             toolchain::Value* offset_elements) {
    return LoadBroadcast(ComputeOffsetPointer(base_pointer, offset_elements));
  }
  toolchain::Value* LoadBroadcast(toolchain::Value* base_pointer,
                             int64_t offset_elements) {
    return LoadBroadcast(base_pointer, b()->getInt64(offset_elements));
  }

  // Compute the horizontal sum of each vector in `vectors`.  The i'th element
  // in the result vector is the (scalar) horizontal sum of the i'th vector in
  // `vectors`.  If `init_values` is not nullptr then the value in the i'th lane
  // in `init_values` is added to the i'th horizontal sum.
  std::vector<toolchain::Value*> ComputeHorizontalSums(
      std::vector<toolchain::Value*> vectors, toolchain::Value* init_values = nullptr);

  toolchain::Value* GetZeroVector();
  toolchain::Value* GetZeroScalar();

  toolchain::IRBuilderBase* b() const { return b_; }
  int64_t vector_size() const { return vector_size_; }
  toolchain::Type* vector_type() const { return vector_type_; }
  toolchain::Type* vector_pointer_type() const { return vector_pointer_type_; }
  toolchain::Type* scalar_type() const { return scalar_type_; }
  toolchain::Type* scalar_pointer_type() const { return scalar_pointer_type_; }
  int64_t scalar_byte_size() const {
    return primitive_util::BitWidth(primitive_type_) / 8;
  }

  const std::string& name() const { return name_; }

 private:
  toolchain::Value* ExtractLowHalf(toolchain::Value*);
  toolchain::Value* ExtractHighHalf(toolchain::Value*);

  toolchain::Value* MulInternal(toolchain::Value* lhs, toolchain::Value* rhs);
  toolchain::Value* AddInternal(toolchain::Value* lhs, toolchain::Value* rhs);

  toolchain::Value* AddReduce(toolchain::Value* vector);

  // Checks that each value in `values` is either of type scalar_type() or
  // vector_type().  This LOG(FATAL)'s so it should only be called in cases
  // where a mismatching type is a programmer bug.
  void AssertCorrectTypes(std::initializer_list<toolchain::Value*> values);

  // Perform an X86 AVX style horizontal add between `lhs` and `rhs`.  The
  // resulting IR for an 8-float wide vector is expected to lower to a single
  // vhaddps instruction on a CPU that supports vhaddps, and not be too bad in
  // other cases.
  //
  // For a vector width of 8, the result vector is computed as:
  //   Result[0] = Lhs[0] + Lhs[1]
  //   Result[1] = Lhs[2] + Lhs[3]
  //   Result[2] = Rhs[0] + Rhs[1]
  //   Result[3] = Rhs[2] + Rhs[3]
  //   Result[4] = Lhs[4] + Lhs[5]
  //   Result[5] = Lhs[6] + Lhs[7]
  //   Result[6] = Rhs[4] + Rhs[5]
  //   Result[7] = Rhs[6] + Rhs[7]
  toolchain::Value* AvxStyleHorizontalAdd(toolchain::Value* lhs, toolchain::Value* rhs);

  std::vector<toolchain::Value*> ComputeAvxOptimizedHorizontalSums(
      std::vector<toolchain::Value*> vectors, toolchain::Value* init_values);

  toolchain::Type* IntegerTypeForFloatSize(bool vector);
  toolchain::Value* I1ToFloat(toolchain::Value* i1);
  toolchain::Value* GetConstantFloat(toolchain::Type* type, const toolchain::APFloat& f) {
    toolchain::Constant* scalar_value = toolchain::ConstantFP::get(type->getContext(), f);
    if (toolchain::isa<toolchain::VectorType>(type)) {
      return toolchain::ConstantVector::getSplat(
          toolchain::ElementCount::getFixed(vector_size()), scalar_value);
    }
    return scalar_value;
  }

  int64_t vector_size_;
  PrimitiveType primitive_type_;
  toolchain::IRBuilderBase* b_;
  toolchain::Type* vector_type_;
  toolchain::Type* vector_pointer_type_;
  toolchain::Type* scalar_type_;
  toolchain::Type* scalar_pointer_type_;
  std::string name_;
};

// This wraps an alloca-backed stack variable which LLVM's SSA construction pass
// can later convert to a SSA value.
class LlvmVariable {
 public:
  LlvmVariable(toolchain::Type*, toolchain::IRBuilderBase* b);

  toolchain::Value* Get() const;
  void Set(toolchain::Value* new_value);

 private:
  toolchain::AllocaInst* alloca_;
  toolchain::IRBuilderBase* b_;
};

class VectorVariable : public LlvmVariable {
 public:
  VectorVariable(VectorIrBuilder* vector_support, toolchain::Value* initial_value)
      : LlvmVariable(vector_support->vector_type(), vector_support->b()) {
    Set(initial_value);
  }
};

class ScalarVariable : public LlvmVariable {
 public:
  ScalarVariable(VectorIrBuilder* vector_support, toolchain::Value* initial_value)
      : LlvmVariable(vector_support->scalar_type(), vector_support->b()) {
    Set(initial_value);
  }
};

// This wraps a set of alloca-backed stack variables that can, as a whole, store
// a tile.  A "tile" is a sequence of vectors that is typically used as a 2D
// grid of scalar values (e.g. for tiled GEMMs).
class TileVariable {
 public:
  TileVariable(VectorIrBuilder* vector_support,
               std::vector<toolchain::Value*> initial_value);

  std::vector<toolchain::Value*> Get() const;
  void Set(absl::Span<toolchain::Value* const> value);

 private:
  std::vector<VectorVariable> storage_;
};

}  // namespace xla::cpu

#endif  // MACHINA_MACHINA_XLA_BACKENDS_CPU_CODEGEN_VECTOR_IR_BUILDER_H_
