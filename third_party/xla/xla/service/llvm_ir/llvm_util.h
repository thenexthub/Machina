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

#ifndef MACHINA_MACHINA_XLA_SERVICE_LLVM_IR_LLVM_UTIL_H_
#define MACHINA_MACHINA_XLA_SERVICE_LLVM_IR_LLVM_UTIL_H_

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "toolchain/IR/BasicBlock.h"
#include "toolchain/IR/FPEnv.h"
#include "toolchain/IR/GlobalVariable.h"
#include "toolchain/IR/IRBuilder.h"
#include "toolchain/IR/InstrTypes.h"
#include "toolchain/IR/Instructions.h"
#include "toolchain/IR/LLVMContext.h"
#include "toolchain/IR/Module.h"
#include "toolchain/IR/Value.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "machina/xla/hlo/ir/hlo_instruction.h"
#include "machina/xla/literal.h"
#include "machina/xla/service/hlo_module_config.h"
#include "machina/xla/shape.h"
#include "machina/xla/xla_data.pb.h"

namespace toolchain {
class FastMathFlags;
class TargetOptions;
};  // namespace toolchain

namespace xla {
namespace llvm_ir {

// We have different DumpToString functions for each type for findability. We
// use pointers / values based on the usual semantics of the parameter type.

std::string DumpToString(const toolchain::Module* module);
std::string DumpToString(const toolchain::Type* type);
std::string DumpToString(const toolchain::Value* value);

// This also works for mlir::Op<...> descendants, such as mlir::ModuleOp.
//
// For findability:
//   std::string DumpToString(mlir::Op<...>& op);
//   std::string DumpToString(mlir::ModuleOp& module_op);
//
// The `operation` parameter is not const, because the used print() method is
// not const.
std::string DumpToString(mlir::Operation* operation);
std::string DumpToString(mlir::Type type);
std::string DumpToString(mlir::Value value);

// Constructs a human-friendly name from the given inputs.  The result is
// suitable for use as an toolchain::Value's name.
//
// This is equivalent to
//
//   - changing the HloInstruction* to its name() (if we called that overload),
//   - joining all of the nonempty inputs by '.', and then
//   - removing all '%'s.
//
std::string IrName(absl::string_view a);
std::string IrName(absl::string_view a, absl::string_view b);
std::string IrName(const HloInstruction* a, absl::string_view b = "");

// Construct a module from the given location with an optional name.
//
// The underlying "create" method is unsafe, because it leaks the new module by
// default. This function avoids this by always returning an OwningOpRef.
mlir::OwningOpRef<mlir::ModuleOp> CreateMlirModuleOp(
    mlir::Location loc, std::optional<toolchain::StringRef> name = std::nullopt);

// Removes special characters from a function name.
//
// Note that this can cause different inputs to map to the same output, so after
// sanitizing a function name, you must run it through a uniquer.
std::string SanitizeFunctionName(std::string function_name);

// Emits a call to the specified intrinsic with the given operands. Overloaded
// intrinsics (for example, "minnum") must include a type in overloaded_types
// for each overloaded type. Typically, overloaded intrinsics have only a single
// overloaded type.
toolchain::CallInst* EmitCallToIntrinsic(
    toolchain::Intrinsic::ID intrinsic_id, absl::Span<toolchain::Value* const> operands,
    absl::Span<toolchain::Type* const> overloaded_types, toolchain::IRBuilderBase* b,
    absl::string_view name = "");

// Emit float max. Emit maxnum intrinsic is fast math is disabled, or
// fcmp+select otherwise
toolchain::Value* EmitFloatMax(toolchain::Value* lhs_value, toolchain::Value* rhs_value,
                          toolchain::IRBuilderBase* b, bool enable_fast_min_max,
                          absl::string_view name = "");

// Emit float min. Emit minnum intrinsic is fast math is disabled, or
// fcmp+select otherwise
toolchain::Value* EmitFloatMin(toolchain::Value* lhs_value, toolchain::Value* rhs_value,
                          toolchain::IRBuilderBase* b, bool enable_fast_min_max,
                          absl::string_view name = "");

// Convenience methods for emitting a GEP instruction that indexes into a buffer
// (1-dimensional array), equivalent to array[index]. The element type of the
// array must be explicitly passed in.  The int64_t index overload
// wraps the index in a i64 toolchain::Value.
toolchain::Value* EmitBufferIndexingGEP(toolchain::Value* array, toolchain::Type* element_type,
                                   toolchain::Value* index, toolchain::IRBuilderBase* b);
toolchain::Value* EmitBufferIndexingGEP(toolchain::Value* array, toolchain::Type* element_type,
                                   int64_t index, toolchain::IRBuilderBase* b);

// Returns the LLVM type which represents the given XLA primitive type.
toolchain::Type* PrimitiveTypeToIrType(PrimitiveType element_type,
                                  toolchain::LLVMContext& context);

// Returns the XLA primitive type which represents the given LLVM type.
// If `default_to_signed_for_integers` is true, then integer types will be
// treated as signed if they are not explicitly specified as unsigned.
PrimitiveType PrimitiveTypeFromIrType(
    toolchain::Type* type, bool default_to_signed_for_integers = true);

// Returns the type size in bits. If "type" is a struct, it must be packed.
int GetSizeInBits(toolchain::Type* type);

// Returns the LLVM type which represents the given XLA shape. For example,
// if "shape" is [5 x [10 x f32]], the function returns [5 x [10 x float]].
toolchain::Type* ShapeToIrType(const Shape& shape, toolchain::LLVMContext& context);

// Returns a value that represents a pointer to a global string constant that
// encodes the shape as a serialized protobuf.
absl::StatusOr<toolchain::Value*> EncodeSelfDescribingShapeConstant(
    const Shape& shape, int32_t* shape_size, toolchain::IRBuilderBase* b);

// Converts a given literal to an IR Constant. Literals have known constant
// values at IR emission time.
toolchain::Constant* ConvertLiteralToIrConstant(const Literal& literal,
                                           toolchain::Module* module);

// Allocates a tile of shared memory.
toolchain::GlobalVariable* AllocateSharedMemoryTile(toolchain::Module* module,
                                               toolchain::Type* tile_type,
                                               absl::string_view name);

// Utility class for working with shared memory.
class SharedMemoryTile {
 public:
  SharedMemoryTile() = default;
  explicit SharedMemoryTile(toolchain::GlobalVariable* base_ptr,
                            toolchain::Type* element_type)
      : base_ptr_(base_ptr), element_type_(element_type) {}

  toolchain::Value* Address(absl::Span<toolchain::Value* const> index,
                       toolchain::IRBuilderBase* b) const;
  toolchain::Value* Load(absl::Span<toolchain::Value* const> index,
                    toolchain::IRBuilderBase* b) const;
  toolchain::StoreInst* Store(toolchain::Value* value,
                         absl::Span<toolchain::Value* const> index,
                         toolchain::IRBuilderBase* b) const;
  toolchain::Type* GetElementType() const { return element_type_; }

 private:
  toolchain::GlobalVariable* base_ptr_;
  toolchain::Type* element_type_;
};

SharedMemoryTile AllocateSharedMemoryTile(
    toolchain::Module* module, toolchain::Type* element_type,
    absl::Span<int64_t const> dimensions_major_to_minor,
    absl::string_view buffer_name);

// Inserts an allocate of the requested type at the entry point of the
// function that the builder is currently building. The insert point
// of the builder is set to the same place after calling this function
// as before.
//
// This can be useful to avoid e.g. executing an alloca every time
// through a loop.
toolchain::AllocaInst* EmitAllocaAtFunctionEntry(toolchain::Type* type,
                                            absl::string_view name,
                                            toolchain::IRBuilderBase* b,
                                            int alignment = 0);

// As EmitAllocaAtFunctionEntry, but allocates element_count entries
// instead of a single element.
toolchain::AllocaInst* EmitAllocaAtFunctionEntryWithCount(toolchain::Type* type,
                                                     toolchain::Value* element_count,
                                                     absl::string_view name,
                                                     toolchain::IRBuilderBase* b,
                                                     int alignment = 0);

// Creates a basic block with the same context and function as for the
// builder. Inserts at the end of the function if insert_before is
// null.
toolchain::BasicBlock* CreateBasicBlock(toolchain::BasicBlock* insert_before,
                                   absl::string_view name,
                                   toolchain::IRBuilderBase* b);

// Struct with data on a conditional branch in a diamond shape created
// via EmitIfThenElse.
struct LlvmIfData {
  // The block that has the conditional branch.
  toolchain::BasicBlock* if_block;

  // The block that is executed if the condition is true.
  toolchain::BasicBlock* true_block;

  // The block that is executed if the condition is false.
  toolchain::BasicBlock* false_block;

  // The block that follows after both the true_block and the
  // false_block.
  toolchain::BasicBlock* after_block;
};

// Inserts a diamond-shaped if-then-else construct at the current
// insertion point of the builder. This involves splitting the current
// block into two blocks, at the insertion point, and introducing a
// true-block and a false-block that connect the two split pieces. The
// true-block is executed if the condition parameter evaluates to true
// and otherwise the false-block is executed. If `emit_else` is false,
// it jumps to the after-block rather than the false-block if the
// condition is false, and the returned `false_block` is null.
//
// Currently the insertion point of the builder must be a well-formed
// block with a terminator. If you need to use this for a
// non-terminated block, just make the function able to do that too.
LlvmIfData EmitIfThenElse(toolchain::Value* condition, absl::string_view name,
                          toolchain::IRBuilderBase* b, bool emit_else = true);

// Emits a compare operation between "lhs" and "rhs" with the given predicate,
// and then converts the result to i8 so that it is addressable.
toolchain::Value* EmitComparison(toolchain::CmpInst::Predicate predicate,
                            toolchain::Value* lhs, toolchain::Value* rhs,
                            toolchain::IRBuilderBase* b,
                            absl::string_view name = "");

// Emits a call that logs the given value with the given tag as a prefix.
// The provided tag and value are passed to a runtime logging call that is
// embedded in this translation unit when the emitted code is executed.
//
// This can be very useful for debugging generated programs in short order when
// developing new generated routines.
//
// Precondition: value must be an int64_t.
// Precondition: tag must be a stable pointer for the lifetime of the generated
// program (the constant pointer is burned in to the program).
void EmitLogging(const char* tag, toolchain::Value* value, toolchain::IRBuilderBase* b);

// Adds alignment metadata to a load instruction using the given alignment.
// The alignment refers to the result of the load, not the load itself.
void SetAlignmentMetadataForLoad(toolchain::LoadInst* load, uint64_t alignment);

// Adds dereferenceable metadata to a load instruction using the given
// the number of dereferenceable bytes.
// Dereferenceable refers to the result of the load, not the load itself.
void SetDereferenceableMetadataForLoad(toolchain::LoadInst* load,
                                       uint64_t dereferenceable_bytes);

// Tells LLVM `inst >= lower && inst < upper`. Returns `inst` for convenience.
toolchain::Instruction* AddRangeMetadata(int32_t lower, int32_t upper,
                                    toolchain::Instruction* inst,
                                    toolchain::Module* module);

void SetToFirstInsertPoint(toolchain::BasicBlock* blk, toolchain::IRBuilderBase* builder);

void SetToLastInsertPoint(toolchain::BasicBlock* blk, toolchain::IRBuilderBase* builder);

// Returns the number of bytes within the shape.
int64_t ByteSizeOf(const Shape& shape, const toolchain::DataLayout& data_layout);

// Gets an toolchain::FastMathFlags that reflects the settings in the given
// module config.
toolchain::FastMathFlags GetCpuFastMathFlags(const HloModuleConfig& module_config);

// Computes a conservative union of the metadata in "a" and "b".  For
// aliasing-related metadata, this means the result can be applied to
// instructions whose aliasing relationship can be described either by "a" *or*
// by "b".
std::map<int, toolchain::MDNode*> MergeMetadata(
    toolchain::LLVMContext* context, const std::map<int, toolchain::MDNode*>& a,
    const std::map<int, toolchain::MDNode*>& b);

// Dumps out `llvm_module` to the path specified in DebugOptions, if dumping is
// enabled for the given HLO module.
//
// A sanitized version of `hlo_module_name` is incorporated into the file name.
// If `optimized` is true then a suffix of "-with-opt.ll" is used, else a suffix
// of "-no-opt.ll" is used.
void DumpIrIfEnabled(const HloModule& hlo_module,
                     const toolchain::Module& llvm_module, bool optimized,
                     absl::string_view filename_suffix = "");

toolchain::Function* CreateCpuFunction(toolchain::FunctionType* function_type,
                                  toolchain::GlobalValue::LinkageTypes linkage,
                                  const HloModuleConfig& module_config,
                                  absl::string_view name, toolchain::Module* module);

// Checks whether a global variable is already created to represent the state
// of a random number generator. If not, creates such a variable. Returns the
// global variable.
toolchain::GlobalVariable* GetOrCreateVariableRngState(toolchain::Module* module,
                                                  toolchain::IRBuilderBase* b);

// Adds a delta value to the global state variable and return the old value of
// the variable.
toolchain::Value* RngGetAndUpdateState(uint64_t delta, toolchain::Module* module,
                                  toolchain::IRBuilderBase* b);

// Gets the LLVM address space that should be used for global variables (e.g.
// XLA's rng state).
unsigned GetGlobalMemoryAddressSpace();

// Emits a block which does "return void". Leaves the insert point as is.
toolchain::BasicBlock* EmitReturnBlock(toolchain::IRBuilderBase* b);

// Emits `if (condition) return`. Assumes that the current function returns
// void.
//
// Can either use a supplied `return_block`, or generate a new one.
void EmitEarlyReturn(toolchain::Value* condition, toolchain::IRBuilderBase* b,
                     toolchain::BasicBlock* return_block = nullptr);

}  // namespace llvm_ir
}  // namespace xla

#endif  // MACHINA_MACHINA_XLA_SERVICE_LLVM_IR_LLVM_UTIL_H_
