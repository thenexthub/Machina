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

#ifndef MACHINA_MACHINA_XLA_SERVICE_ELEMENTAL_IR_EMITTER_H_
#define MACHINA_MACHINA_XLA_SERVICE_ELEMENTAL_IR_EMITTER_H_

#include <tuple>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "toolchain/ADT/ArrayRef.h"
#include "toolchain/IR/IRBuilder.h"
#include "toolchain/IR/Module.h"
#include "toolchain/IR/Value.h"
#include "machina/xla/hlo/ir/hlo_instruction.h"
#include "machina/xla/hlo/ir/hlo_instructions.h"
#include "machina/xla/service/llvm_ir/ir_array.h"
#include "machina/xla/service/llvm_ir/ir_builder_mixin.h"
#include "machina/xla/service/llvm_ir/loop_emitter.h"

namespace xla {

class ElementalIrEmitter : public IrBuilderMixin<ElementalIrEmitter> {
 public:
  struct Options {
    // Instead of relying on builtin `fpext` and `fpcast` emit a bitcast and
    // truncate to convert f32 to bf16 (and emit extend to convert bf16 to f32).
    bool xla_cpu_use_truncate_f32_to_bf16_conversion = false;
  };

  using HloToElementGeneratorMap =
      absl::flat_hash_map<const HloInstruction*, llvm_ir::ElementGenerator>;

  ElementalIrEmitter(toolchain::Module* module, toolchain::IRBuilderBase* b,
                     const Options& options)
      : b_(b), module_(module), options_(options) {}

  ElementalIrEmitter(toolchain::Module* module, toolchain::IRBuilderBase* b)
      : ElementalIrEmitter(module, b, Options()) {}

  virtual ~ElementalIrEmitter() = default;

  // Returns a function to generate an element of the output of `hlo`, given a
  // map of functions to generate elements of its operands.
  llvm_ir::ElementGenerator MakeElementGenerator(
      const HloInstruction* hlo,
      const HloToElementGeneratorMap& operand_to_generator);

  toolchain::IRBuilderBase* b() { return b_; }

  // builder() is for IrBuilderMixin.
  toolchain::IRBuilderBase* builder() { return b_; }

  toolchain::Module* module() { return module_; }

  // Returns which ops invalidate the cache of emitted instructions by creating
  // a new BasicBlock and setting the insertion point to the newly created
  // BasicBlock. We can only reuse cached values if they were emitted in the
  // same BasicBlock as the current BasicBlock.
  static bool OpInvalidatesCache(const HloInstruction* hlo);

 protected:
  virtual llvm_ir::IrArray::Index GetSourceIndexOfBitcast(
      const llvm_ir::IrArray::Index& index, const HloInstruction* hlo) {
    return index.SourceIndexOfBitcast(hlo->shape(), hlo->operand(0)->shape(),
                                      b_);
  }

  virtual absl::StatusOr<toolchain::Value*> EmitFloatBinaryOp(
      const HloInstruction* op, toolchain::Value* lhs_value, toolchain::Value* rhs_value);

  virtual toolchain::Value* EmitExtractReal(toolchain::Value* value);
  virtual toolchain::Value* EmitExtractImag(toolchain::Value* value);

 private:
  virtual absl::StatusOr<toolchain::Value*> EmitUnaryOp(const HloInstruction* op,
                                                   toolchain::Value* operand_value);

  virtual absl::StatusOr<toolchain::Value*> EmitBinaryOp(const HloInstruction* op,
                                                    toolchain::Value* lhs_value,
                                                    toolchain::Value* rhs_value);

  virtual absl::StatusOr<toolchain::Value*> EmitIntegerUnaryOp(
      const HloInstruction* op, toolchain::Value* operand_value);

  virtual absl::StatusOr<toolchain::Value*> EmitFloatUnaryOp(
      const HloInstruction* op, toolchain::Value* operand_value);

  virtual absl::StatusOr<toolchain::Value*> EmitComplexUnaryOp(
      const HloInstruction* op, toolchain::Value* operand_value);

  toolchain::Value* IsZero(toolchain::Value* v);
  toolchain::Value* IsIntMinDivisionOverflow(toolchain::Value* lhs, toolchain::Value* rhs);
  toolchain::Value* GetZero(toolchain::Type* type);
  toolchain::Value* GetOne(toolchain::Type* type);
  toolchain::Value* GetIntSMin(toolchain::Type* type);
  toolchain::Value* GetMinusOne(toolchain::Type* type);

  toolchain::Value* EmitIntegerDivide(toolchain::Value* lhs, toolchain::Value* rhs,
                                 bool is_signed);
  toolchain::Value* EmitIntegerRemainder(toolchain::Value* lhs, toolchain::Value* rhs,
                                    bool is_signed);
  toolchain::Value* EmitIntegerPow(toolchain::Value* lhs, toolchain::Value* rhs,
                              bool is_signed);

  virtual absl::StatusOr<toolchain::Value*> EmitPredBinaryOp(
      const HloInstruction* op, toolchain::Value* lhs_value, toolchain::Value* rhs_value);

  virtual absl::StatusOr<toolchain::Value*> EmitIntegerBinaryOp(
      const HloInstruction* op, toolchain::Value* lhs_value, toolchain::Value* rhs_value,
      bool is_signed);

  virtual absl::StatusOr<toolchain::Value*> EmitComplexBinaryOp(
      const HloInstruction* op, toolchain::Value* lhs_value, toolchain::Value* rhs_value);

  virtual toolchain::Value* EmitFloatMax(toolchain::Value* lhs_value,
                                    toolchain::Value* rhs_value,
                                    absl::string_view name);

  virtual toolchain::Value* EmitFloatMin(toolchain::Value* lhs_value,
                                    toolchain::Value* rhs_value,
                                    absl::string_view name);

  toolchain::Value* EmitIntegralMax(toolchain::Value* lhs_value, toolchain::Value* rhs_value,
                               bool is_signed);

  toolchain::Value* EmitIntegralMin(toolchain::Value* lhs_value, toolchain::Value* rhs_value,
                               bool is_signed);

  virtual absl::StatusOr<toolchain::Value*> EmitAtan2(PrimitiveType prim_type,
                                                 toolchain::Value* lhs,
                                                 toolchain::Value* rhs,
                                                 absl::string_view name);

  virtual absl::StatusOr<toolchain::Value*> EmitLog(PrimitiveType prim_type,
                                               toolchain::Value* value);

  virtual absl::StatusOr<toolchain::Value*> EmitSqrt(PrimitiveType prim_type,
                                                toolchain::Value* value);

  virtual absl::StatusOr<toolchain::Value*> EmitCbrt(PrimitiveType prim_type,
                                                toolchain::Value* value);

  virtual absl::StatusOr<toolchain::Value*> EmitRsqrt(PrimitiveType prim_type,
                                                 toolchain::Value* value);

  virtual absl::StatusOr<toolchain::Value*> EmitLog1p(PrimitiveType prim_type,
                                                 toolchain::Value* value);

  virtual absl::StatusOr<toolchain::Value*> EmitSin(PrimitiveType prim_type,
                                               toolchain::Value* value);

  virtual absl::StatusOr<toolchain::Value*> EmitCos(PrimitiveType prim_type,
                                               toolchain::Value* value);

  virtual absl::StatusOr<toolchain::Value*> EmitCosm1(PrimitiveType prim_type,
                                                 toolchain::Value* value);

  virtual absl::StatusOr<toolchain::Value*> EmitTan(PrimitiveType prim_type,
                                               toolchain::Value* value);

  virtual absl::StatusOr<toolchain::Value*> EmitExp(PrimitiveType prim_type,
                                               toolchain::Value* value,
                                               absl::string_view name);

  virtual absl::StatusOr<toolchain::Value*> EmitExpm1(PrimitiveType prim_type,
                                                 toolchain::Value* value);

  virtual absl::StatusOr<toolchain::Value*> EmitPow(PrimitiveType prim_type,
                                               toolchain::Value* lhs,
                                               toolchain::Value* rhs,
                                               absl::string_view name);

  virtual absl::StatusOr<toolchain::Value*> EmitErf(PrimitiveType prim_type,
                                               toolchain::Value* value);

  virtual absl::StatusOr<toolchain::Value*> EmitTanh(PrimitiveType prim_type,
                                                toolchain::Value* value);

  virtual absl::StatusOr<toolchain::Value*> EmitReducePrecision(
      const HloInstruction* hlo, toolchain::Value* x);

  virtual absl::StatusOr<std::tuple<toolchain::Value*, toolchain::Value*, toolchain::Value*>>
  EmitComplexAbsHelper(PrimitiveType prim_type, toolchain::Value* real,
                       toolchain::Value* imag, bool return_sqrt);

  virtual absl::StatusOr<toolchain::Value*> EmitComplexAbs(
      PrimitiveType prim_type, toolchain::Value* operand_value);

  virtual absl::StatusOr<toolchain::Value*> EmitSqrtComplexAbs(
      PrimitiveType prim_type, toolchain::Value* operand_value);
  virtual absl::StatusOr<toolchain::Value*> EmitRsqrtComplexAbs(
      PrimitiveType prim_type, toolchain::Value* operand_value);

  virtual absl::StatusOr<toolchain::Value*> EmitComplexAdd(const HloInstruction* op,
                                                      toolchain::Value* lhs_value,
                                                      toolchain::Value* rhs_value);

  virtual absl::StatusOr<toolchain::Value*> EmitComplexSubtract(
      const HloInstruction* op, toolchain::Value* lhs_value, toolchain::Value* rhs_value);

  virtual absl::StatusOr<toolchain::Value*> EmitComplexMultiply(
      const HloInstruction* op, toolchain::Value* lhs_value, toolchain::Value* rhs_value);

  virtual absl::StatusOr<toolchain::Value*> EmitComplexDivide(
      const HloInstruction* op, toolchain::Value* lhs_value, toolchain::Value* rhs_value);

  virtual absl::StatusOr<toolchain::Value*> EmitComplexLog(
      const HloInstruction* op, toolchain::Value* operand_value);

  virtual absl::StatusOr<toolchain::Value*> EmitComplexSqrt(
      const HloInstruction* op, PrimitiveType prim_type,
      toolchain::Value* operand_value);

  virtual absl::StatusOr<toolchain::Value*> EmitComplexRsqrt(
      const HloInstruction* op, PrimitiveType prim_type,
      toolchain::Value* operand_value);

  absl::StatusOr<toolchain::Value*> EmitAccumResult(
      absl::Span<toolchain::Value* const> accumulator_addrs,
      toolchain::ArrayRef<toolchain::Type*> accumulator_types, bool is_variadic);

  // Composes a complex struct. imag may be nullptr for simple cast operations.
  toolchain::Value* EmitComposeComplex(const HloInstruction* op, toolchain::Value* real,
                                  toolchain::Value* imag);

  // Emit `accumulator + lhs * rhs` for the given primitive type.
  toolchain::Value* EmitMulAdd(toolchain::Value* lhs, toolchain::Value* rhs,
                          toolchain::Value* accumulator,
                          xla::PrimitiveType primitive_type);

  absl::StatusOr<toolchain::Value*> EmitElementalSelect(
      const HloInstruction* hlo,
      const HloToElementGeneratorMap& operand_to_generator,
      const llvm_ir::IrArray::Index& index);

  absl::StatusOr<toolchain::Value*> EmitElementalClamp(
      const HloInstruction* hlo,
      const HloToElementGeneratorMap& operand_to_generator,
      const llvm_ir::IrArray::Index& index);

  absl::StatusOr<toolchain::Value*> EmitElementalConcatenate(
      const HloInstruction* hlo,
      const HloToElementGeneratorMap& operand_to_generator,
      const llvm_ir::IrArray::Index& target_index);

  absl::StatusOr<toolchain::Value*> EmitElementalDynamicSlice(
      const HloInstruction* hlo,
      const HloToElementGeneratorMap& operand_to_generator,
      const llvm_ir::IrArray::Index& index);

  absl::StatusOr<toolchain::Value*> EmitElementalGather(
      const HloInstruction* hlo,
      const HloToElementGeneratorMap& operand_to_generator,
      const llvm_ir::IrArray::Index& index);

  absl::StatusOr<toolchain::Value*> EmitElementalDynamicUpdateSlice(
      const HloInstruction* hlo,
      const HloToElementGeneratorMap& operand_to_generator,
      const llvm_ir::IrArray::Index& index);

  absl::StatusOr<toolchain::Value*> EmitElementalPad(
      const HloInstruction* hlo,
      const HloToElementGeneratorMap& operand_to_generator,
      const llvm_ir::IrArray::Index& padded_index);

  absl::StatusOr<toolchain::Value*> EmitElementalDot(
      const HloInstruction* hlo,
      const HloToElementGeneratorMap& operand_to_generator,
      const llvm_ir::IrArray::Index& dot_result_index);

  virtual absl::StatusOr<std::vector<toolchain::Value*>> EmitThreadLocalCall(
      const HloComputation& callee, absl::Span<toolchain::Value* const> parameters,
      absl::string_view name, bool is_reducer);

  absl::StatusOr<toolchain::Value*> EmitElementalMap(
      const HloMapInstruction* map_instr,
      absl::Span<toolchain::Value* const> elemental_operands);

  absl::StatusOr<toolchain::Value*> EmitElementalReduceWindow(
      const HloReduceWindowInstruction* reduce_window,
      std::vector<llvm_ir::ElementGenerator> input_generators,
      std::vector<llvm_ir::ElementGenerator> initial_value_generators,
      const llvm_ir::IrArray::Index& index);

  absl::StatusOr<toolchain::Value*> EmitElementalReduce(
      const HloReduceInstruction* reduce,
      std::vector<llvm_ir::ElementGenerator> input_generators,
      std::vector<llvm_ir::ElementGenerator> initial_value_generators,
      const llvm_ir::IrArray::Index& index);

  virtual absl::StatusOr<toolchain::Value*> EmitConvolution(
      const HloInstruction* hlo,
      const HloToElementGeneratorMap& operand_to_generator,
      const llvm_ir::IrArray::Index& index);

  // Computes the complex power function.
  absl::StatusOr<toolchain::Value*> EmitComplexPower(const HloInstruction* op,
                                                toolchain::Value* lhs_value,
                                                toolchain::Value* rhs_value);

  // Evaluates a polynomial using Horner's method.
  absl::StatusOr<toolchain::Value*> EvaluatePolynomial(
      toolchain::Type* type, toolchain::Value* x, absl::Span<const double> coefficients);

  virtual bool fast_min_max();

  toolchain::IRBuilderBase* const b_;

  toolchain::Module* module_;

  Options options_;

  friend class ElementalIrEmitterForTests;
};

// Allow to instantiate IR emitter in tests.
class ElementalIrEmitterForTests : public ElementalIrEmitter {
 public:
  ElementalIrEmitterForTests(toolchain::Module* module, toolchain::IRBuilderBase* builder)
      : ElementalIrEmitter(module, builder) {}

  absl::Status TestElementalDot(const HloInstruction* hlo,
                                const llvm_ir::IrArray::Index& index) {
    return EmitElementalDot(hlo, generator_map_, index).status();
  }

 private:
  absl::StatusOr<std::vector<toolchain::Value*>> EmitThreadLocalCall(
      const HloComputation& callee, absl::Span<toolchain::Value* const> parameters,
      absl::string_view name, bool is_reducer) override {
    return absl::UnimplementedError("");
  }
  bool fast_min_max() override { return false; }

  HloToElementGeneratorMap generator_map_;
};

absl::StatusOr<toolchain::Value*> EmitReducePrecisionIR(
    PrimitiveType src_ty, toolchain::Value* x, int64_t dest_exponent_bits,
    int64_t dest_mantissa_bits, bool quiet_nans, toolchain::IRBuilderBase* b);

}  // namespace xla

#endif  // MACHINA_MACHINA_XLA_SERVICE_ELEMENTAL_IR_EMITTER_H_
