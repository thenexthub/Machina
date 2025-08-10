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

#include "machina/xla/service/llvm_ir/llvm_util.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/casts.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "toolchain/ADT/ArrayRef.h"
#include "toolchain/ADT/STLExtras.h"
#include "toolchain/ADT/SmallPtrSet.h"
#include "toolchain/ADT/SmallVector.h"
#include "toolchain/ADT/StringRef.h"
#include "toolchain/IR/Attributes.h"
#include "toolchain/IR/CallingConv.h"
#include "toolchain/IR/Constants.h"
#include "toolchain/IR/DataLayout.h"
#include "toolchain/IR/DerivedTypes.h"
#include "toolchain/IR/GlobalValue.h"
#include "toolchain/IR/GlobalVariable.h"
#include "toolchain/IR/IRBuilder.h"
#include "toolchain/IR/InstrTypes.h"
#include "toolchain/IR/Instructions.h"
#include "toolchain/IR/Intrinsics.h"
#include "toolchain/IR/LLVMContext.h"
#include "toolchain/IR/MDBuilder.h"
#include "toolchain/IR/Type.h"
#include "toolchain/Support/Alignment.h"
#include "toolchain/Support/Casting.h"
#include "toolchain/Support/CodeGen.h"
#include "toolchain/Support/raw_ostream.h"
#include "toolchain/TargetParser/Triple.h"
#include "toolchain/Transforms/Utils/Cloning.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "machina/xla/layout_util.h"
#include "machina/xla/literal.h"
#include "machina/xla/primitive_util.h"
#include "machina/xla/service/cpu/cpu_options.h"
#include "machina/xla/service/dump.h"
#include "machina/xla/service/hlo_module_config.h"
#include "machina/xla/service/llvm_ir/llvm_type_conversion_util.h"
#include "machina/xla/shape.h"
#include "machina/xla/shape_util.h"
#include "machina/xla/tsl/platform/byte_order.h"
#include "machina/xla/tsl/platform/logging.h"
#include "machina/xla/util.h"
#include "machina/xla/xla_data.pb.h"
#include "tsl/profiler/lib/scoped_annotation.h"

namespace xla {
namespace llvm_ir {

namespace {

// This works for most toolchain / mlir types. This also accepts a const pointer to
// objects which have a const print() method.
template <typename T>
std::string DumpToStringTempl(T* entity) {
  CHECK_NE(entity, nullptr);

  std::string s;
  toolchain::raw_string_ostream ostream(s);
  ostream << *entity;
  return s;
}

// Note, this function is only useful in an insertion context; in a global
// (e.g. constants) context it will CHECK fail.
toolchain::Module* ModuleFromIRBuilder(toolchain::IRBuilderBase* b) {
  auto block = CHECK_NOTNULL(b->GetInsertBlock());
  auto fn = CHECK_NOTNULL(block->getParent());
  auto module = CHECK_NOTNULL(fn->getParent());
  return module;
}

PrimitiveType PrimitiveTypeFromIrIntegerType(
    toolchain::IntegerType* type, bool default_to_signed_for_integers) {
  // PRED (boolean) is typically a 1-bit integer.
  if (type->getBitWidth() == 1) {
    return PRED;
  }

  // LLVM's toolchain::IntegerType (e.g., i8, i32) does not distinguish between
  // signed and unsigned types by itself. The interpretation (signed/unsigned)
  // depends on the operations using these types (e.g., sdiv vs. udiv).
  // The 'default_to_signed_for_integers' flag helps make a choice here.
  switch (type->getBitWidth()) {
    case 8:
      return default_to_signed_for_integers ? S8 : U8;
    case 16:
      return default_to_signed_for_integers ? S16 : U16;
    case 32:
      return default_to_signed_for_integers ? S32 : U32;
    case 64:
      return default_to_signed_for_integers ? S64 : U64;
    default:
      return PRIMITIVE_TYPE_INVALID;
  }
}

std::optional<PrimitiveType> PrimitiveComplexTypeFromIrStructType(
    toolchain::StructType* struct_type) {
  // XLA C64 is typically represented as an LLVM struct {float, float}.
  // XLA C128 is typically represented as an LLVM struct {double, double}.
  if (struct_type->getNumElements() == 2) {
    toolchain::Type* el_type0 = struct_type->getElementType(0);
    toolchain::Type* el_type1 = struct_type->getElementType(1);
    if (el_type0->isFloatTy() && el_type1->isFloatTy()) {
      return C64;  // Complex64
    }
    if (el_type0->isDoubleTy() && el_type1->isDoubleTy()) {
      return C128;  // Complex128
    }
  }
  return std::nullopt;
}

}  // namespace

std::string DumpToString(const toolchain::Module* module) {
  return DumpToStringTempl(module);
}

std::string DumpToString(const toolchain::Type* type) {
  return DumpToStringTempl(type);
}

std::string DumpToString(const toolchain::Value* value) {
  return DumpToStringTempl(value);
}

std::string DumpToString(mlir::Operation* operation) {
  return DumpToStringTempl(operation);
}

std::string DumpToString(mlir::Type type) { return DumpToStringTempl(&type); }

std::string DumpToString(mlir::Value value) {
  return DumpToStringTempl(&value);
}

toolchain::CallInst* EmitCallToIntrinsic(
    toolchain::Intrinsic::ID intrinsic_id, absl::Span<toolchain::Value* const> operands,
    absl::Span<toolchain::Type* const> overloaded_types, toolchain::IRBuilderBase* b,
    absl::string_view name) {
  toolchain::Module* module = ModuleFromIRBuilder(b);
  toolchain::Function* intrinsic = toolchain::Intrinsic::getOrInsertDeclaration(
      module, intrinsic_id, AsArrayRef(overloaded_types));
  return b->CreateCall(intrinsic, AsArrayRef(operands), AsStringRef(name));
}

toolchain::Value* EmitFloatMax(toolchain::Value* lhs_value, toolchain::Value* rhs_value,
                          toolchain::IRBuilderBase* b, bool enable_fast_min_max,
                          absl::string_view name) {
  if (b->getFastMathFlags().noNaNs() || enable_fast_min_max) {
    auto cmp = b->CreateFCmpUGE(lhs_value, rhs_value);
    return b->CreateSelect(cmp, lhs_value, rhs_value, AsStringRef(name));
  }
  return llvm_ir::EmitCallToIntrinsic(toolchain::Intrinsic::maximum,
                                      {lhs_value, rhs_value},
                                      {lhs_value->getType()}, b);
}

toolchain::Value* EmitFloatMin(toolchain::Value* lhs_value, toolchain::Value* rhs_value,
                          toolchain::IRBuilderBase* b, bool enable_fast_min_max,
                          absl::string_view name) {
  if (b->getFastMathFlags().noNaNs() || enable_fast_min_max) {
    auto cmp = b->CreateFCmpULE(lhs_value, rhs_value);
    return b->CreateSelect(cmp, lhs_value, rhs_value, AsStringRef(name));
  }
  return llvm_ir::EmitCallToIntrinsic(toolchain::Intrinsic::minimum,
                                      {lhs_value, rhs_value},
                                      {lhs_value->getType()}, b);
}

toolchain::Value* EmitBufferIndexingGEP(toolchain::Value* array, toolchain::Type* element_type,
                                   toolchain::Value* index, toolchain::IRBuilderBase* b) {
  toolchain::Type* array_type = array->getType();
  CHECK(array_type->isPointerTy());
  VLOG(2) << "EmitBufferIndexingGEP with type="
          << llvm_ir::DumpToString(array_type)
          << " array=" << llvm_ir::DumpToString(array)
          << " index=" << llvm_ir::DumpToString(index);

  return b->CreateInBoundsGEP(
      element_type, array,
      toolchain::isa<toolchain::GlobalVariable>(array)
          ? toolchain::ArrayRef<toolchain::Value*>({b->getInt64(0), index})
          : index);
}

toolchain::Value* EmitBufferIndexingGEP(toolchain::Value* array, toolchain::Type* element_type,
                                   int64_t index, toolchain::IRBuilderBase* b) {
  return EmitBufferIndexingGEP(array, element_type, b->getInt64(index), b);
}

toolchain::Type* PrimitiveTypeToIrType(PrimitiveType element_type,
                                  toolchain::LLVMContext& context) {
  switch (element_type) {
    case S2:
    case U2:
      return toolchain::Type::getIntNTy(context, 2);
    case S4:
    case U4:
      return toolchain::Type::getIntNTy(context, 4);
    case PRED:
    case S8:
    case U8:
      return toolchain::Type::getInt8Ty(context);
    case S16:
    case U16:
      return toolchain::Type::getInt16Ty(context);
    case F4E2M1FN:
      return toolchain::Type::getIntNTy(context, 4);
    case F8E5M2:
    case F8E5M2FNUZ:
    case F8E4M3:
    case F8E4M3FN:
    case F8E4M3B11FNUZ:
    case F8E4M3FNUZ:
    case F8E3M4:
    case F8E8M0FNU:
      // We represent F8 as an int since there is no LLVM F8 dtype.
      return toolchain::Type::getInt8Ty(context);
    case BF16:
      return toolchain::Type::getBFloatTy(context);
    case F16:
      return toolchain::Type::getHalfTy(context);
    case S32:
    case U32:
      return toolchain::Type::getInt32Ty(context);
    case S64:
    case U64:
      return toolchain::Type::getInt64Ty(context);
    case F32:
      return toolchain::Type::getFloatTy(context);
    case F64:
      return toolchain::Type::getDoubleTy(context);
    case C64: {
      auto cplx_t = toolchain::StructType::getTypeByName(context, "complex64");
      if (cplx_t == nullptr) {
        // C++ standard dictates the memory layout of std::complex is contiguous
        // real followed by imaginary. C++11 section 26.4 [complex.numbers]:
        // If z is an lvalue expression of type cv std::complex<T> then the
        // expression reinterpret_cast<cv T(&)[2]>(z) shall be well-formed,
        // reinterpret_cast<cv T(&)[2]>(z)[0] shall designate the real part of
        // z, and reinterpret_cast<cv T(&)[2]>(z)[1] shall designate the
        // imaginary part of z.
        return toolchain::StructType::create(
            {toolchain::Type::getFloatTy(context), toolchain::Type::getFloatTy(context)},
            "complex64", /*isPacked=*/true);
      }
      return cplx_t;
    }
    case C128: {
      auto cplx_t = toolchain::StructType::getTypeByName(context, "complex128");
      if (cplx_t == nullptr) {
        return toolchain::StructType::create({toolchain::Type::getDoubleTy(context),
                                         toolchain::Type::getDoubleTy(context)},
                                        "complex128", /*isPacked=*/true);
      }
      return cplx_t;
    }  // A Tuple contains an array of pointers. Use i8*.
    case TUPLE:
    // An Opaque is like a void*, use i8*.
    case OPAQUE_TYPE:
      return toolchain::PointerType::getUnqual(context);
    case TOKEN:
      // Tokens do not have a physical representation, but the compiler needs
      // some placeholder type, so use int8_t*.
      return toolchain::PointerType::getUnqual(context);
    default:
      LOG(FATAL) << "unsupported type " << element_type;
  }
}

PrimitiveType PrimitiveTypeFromIrType(toolchain::Type* type,
                                      bool default_to_signed_for_integers) {
  if (!type) {
    return PRIMITIVE_TYPE_INVALID;
  }

  // If it's a vector type, XLA PrimitiveType refers to the element type.
  // So, we get the underlying element type for further checks.
  if (type->isVectorTy()) {
    type = toolchain::cast<toolchain::VectorType>(type)->getElementType();
  }

  // Floating-point types
  if (type->isHalfTy()) {
    return F16;
  }
  if (type->isBFloatTy()) {
    return BF16;
  }
  if (type->isFloatTy()) {
    return F32;
  }
  if (type->isDoubleTy()) {
    return F64;
  }

  if (type->isIntegerTy()) {
    return PrimitiveTypeFromIrIntegerType(toolchain::cast<toolchain::IntegerType>(type),
                                          default_to_signed_for_integers);
  }

  if (type->isStructTy()) {
    if (auto result = PrimitiveComplexTypeFromIrStructType(
            toolchain::cast<toolchain::StructType>(type))) {
      return *result;
    }
  }

  if (type->isPointerTy()) {
    return OPAQUE_TYPE;
  }

  return PRIMITIVE_TYPE_INVALID;
}

int GetSizeInBits(toolchain::Type* type) {
  const toolchain::StructType* struct_ty = toolchain::dyn_cast<toolchain::StructType>(type);
  if (struct_ty) {
    CHECK(struct_ty->isPacked());
    int bits = 0;
    for (auto element_type : struct_ty->elements()) {
      bits += GetSizeInBits(element_type);
    }
    return bits;
  }
  int bits = type->getPrimitiveSizeInBits();
  CHECK_GT(bits, 0) << "type is not sized";
  return bits;
}

toolchain::Type* ShapeToIrType(const Shape& shape, toolchain::LLVMContext& context) {
  toolchain::Type* result_type =
      PrimitiveTypeToIrType(shape.element_type(), context);
  if (shape.IsTuple()) {
    // A tuple buffer is an array of pointers.
    result_type =
        toolchain::ArrayType::get(result_type, shape.tuple_shapes().size());
  } else if (shape.IsArray()) {
    for (int64_t dimension : LayoutUtil::MinorToMajor(shape)) {
      result_type =
          toolchain::ArrayType::get(result_type, shape.dimensions(dimension));
    }
  }
  return result_type;
}

absl::StatusOr<toolchain::Value*> EncodeSelfDescribingShapeConstant(
    const Shape& shape, int32_t* shape_size, toolchain::IRBuilderBase* b) {
  const std::string encoded_shape = shape.ToProto().SerializeAsString();
  if (encoded_shape.size() > std::numeric_limits<int32_t>::max()) {
    return Internal("Encoded shape size exceeded int32_t size limit.");
  }
  *shape_size = static_cast<int32_t>(encoded_shape.size());
  return b->CreateGlobalStringPtr(encoded_shape);
}

toolchain::Constant* ConvertLiteralToIrConstant(const Literal& literal,
                                           toolchain::Module* module) {
  const char* data = static_cast<const char*>(literal.untyped_data());
  int64_t size_bytes = literal.size_bytes();
  CHECK_EQ(module->getDataLayout().isLittleEndian(), tsl::port::kLittleEndian);
  std::vector<char> packed_data;
  if (primitive_util::IsSubByteNonPredType(literal.shape().element_type())) {
    auto bit_width = primitive_util::BitWidth(literal.shape().element_type());
    int elements_per_byte = 8 / bit_width;
    packed_data.resize(CeilOfRatio<int64_t>(size_bytes, elements_per_byte));
    PackIntN(bit_width, absl::MakeSpan(data, size_bytes),
             absl::MakeSpan(packed_data));
    data = packed_data.data();
    size_bytes = packed_data.size();
  }
  return toolchain::ConstantDataArray::getString(module->getContext(),
                                            toolchain::StringRef(data, size_bytes),
                                            /*AddNull=*/false);
}

toolchain::GlobalVariable* AllocateSharedMemoryTile(toolchain::Module* module,
                                               toolchain::Type* tile_type,
                                               absl::string_view name) {
  // Both AMDGPU and NVPTX use the same address space for shared memory.
  const int kGPUSharedMemoryAddrSpace = 3;
  return new toolchain::GlobalVariable(
      *module, tile_type,
      /*isConstant=*/false, toolchain::GlobalValue::PrivateLinkage,
      toolchain::UndefValue::get(tile_type), AsStringRef(name), nullptr,
      toolchain::GlobalValue::NotThreadLocal, kGPUSharedMemoryAddrSpace);
}

SharedMemoryTile AllocateSharedMemoryTile(
    toolchain::Module* module, toolchain::Type* element_type,
    absl::Span<int64_t const> dimensions_major_to_minor,
    absl::string_view buffer_name) {
  toolchain::Type* ty = element_type;
  for (auto dim : toolchain::reverse(dimensions_major_to_minor)) {
    ty = toolchain::ArrayType::get(ty, dim);
  }
  return SharedMemoryTile{
      llvm_ir::AllocateSharedMemoryTile(module, ty, buffer_name), element_type};
}

static std::vector<toolchain::Value*> IndexWith0(
    absl::Span<toolchain::Value* const> index, toolchain::IRBuilderBase* b) {
  std::vector<toolchain::Value*> index_with_0{
      toolchain::ConstantInt::get(index.front()->getType(), 0)};
  absl::c_copy(index, std::back_inserter(index_with_0));
  return index_with_0;
}

toolchain::Value* SharedMemoryTile::Address(absl::Span<toolchain::Value* const> index,
                                       toolchain::IRBuilderBase* b) const {
  toolchain::Value* gep = b->CreateInBoundsGEP(base_ptr_->getValueType(), base_ptr_,
                                          IndexWith0(index, b));
  // __shared__ memory uses a different address space, so we cast it
  // to global address space before writing or reading.
  return b->CreateAddrSpaceCast(gep,
                                toolchain::PointerType::get(b->getContext(), 0));
};

toolchain::Value* SharedMemoryTile::Load(absl::Span<toolchain::Value* const> index,
                                    toolchain::IRBuilderBase* b) const {
  auto* load_type = toolchain::GetElementPtrInst::getIndexedType(
      base_ptr_->getValueType(), IndexWith0(index, b));
  return b->CreateLoad(load_type, Address(index, b));
}

toolchain::StoreInst* SharedMemoryTile::Store(toolchain::Value* value,
                                         absl::Span<toolchain::Value* const> index,
                                         toolchain::IRBuilderBase* b) const {
  return b->CreateStore(value, Address(index, b));
}

toolchain::AllocaInst* EmitAllocaAtFunctionEntry(toolchain::Type* type,
                                            absl::string_view name,
                                            toolchain::IRBuilderBase* b,
                                            int alignment) {
  return EmitAllocaAtFunctionEntryWithCount(type, nullptr, name, b, alignment);
}

toolchain::AllocaInst* EmitAllocaAtFunctionEntryWithCount(toolchain::Type* type,
                                                     toolchain::Value* element_count,
                                                     absl::string_view name,
                                                     toolchain::IRBuilderBase* b,
                                                     int alignment) {
  toolchain::IRBuilderBase::InsertPointGuard guard(*b);
  toolchain::Function* function = b->GetInsertBlock()->getParent();
  b->SetInsertPoint(&function->getEntryBlock(),
                    function->getEntryBlock().getFirstInsertionPt());
  toolchain::Module* module = b->GetInsertBlock()->getModule();
  // Explicitly set local addrspace for SPIR backend.
  toolchain::Triple target(module->getTargetTriple());
  int addrspace = target.isSPIR() || target.isAMDGPU() ? 5 : 0;
  toolchain::AllocaInst* alloca =
      b->CreateAlloca(type, addrspace, element_count, AsStringRef(name));
  if (alignment != 0) {
    alloca->setAlignment(toolchain::Align(alignment));
  }
  return alloca;
}

toolchain::BasicBlock* CreateBasicBlock(toolchain::BasicBlock* insert_before,
                                   absl::string_view name,
                                   toolchain::IRBuilderBase* b) {
  return toolchain::BasicBlock::Create(
      /*Context=*/b->getContext(),
      /*Name=*/AsStringRef(name),
      /*Parent=*/b->GetInsertBlock()->getParent(),
      /*InsertBefore*/ insert_before);
}

LlvmIfData EmitIfThenElse(toolchain::Value* condition, absl::string_view name,
                          toolchain::IRBuilderBase* b, bool emit_else) {
  llvm_ir::LlvmIfData if_data;
  if_data.if_block = b->GetInsertBlock();
  if_data.true_block =
      CreateBasicBlock(nullptr, absl::StrCat(name, "-true"), b);
  if_data.false_block =
      emit_else ? CreateBasicBlock(nullptr, absl::StrCat(name, "-false"), b)
                : nullptr;

  // Add a terminator to the if block, if necessary.
  if (if_data.if_block->getTerminator() == nullptr) {
    b->SetInsertPoint(if_data.if_block);
    if_data.after_block =
        CreateBasicBlock(nullptr, absl::StrCat(name, "-after"), b);
    b->CreateBr(if_data.after_block);
  } else {
    if_data.after_block = if_data.if_block->splitBasicBlock(
        b->GetInsertPoint(), absl::StrCat(name, "-after"));
  }

  // Our basic block should now end with an unconditional branch.  Remove it;
  // we're going to replace it with a conditional branch.
  if_data.if_block->getTerminator()->eraseFromParent();

  b->SetInsertPoint(if_data.if_block);
  b->CreateCondBr(condition, if_data.true_block,
                  emit_else ? if_data.false_block : if_data.after_block);

  b->SetInsertPoint(if_data.true_block);
  b->CreateBr(if_data.after_block);

  if (emit_else) {
    b->SetInsertPoint(if_data.false_block);
    b->CreateBr(if_data.after_block);
  }

  b->SetInsertPoint(if_data.after_block,
                    if_data.after_block->getFirstInsertionPt());

  return if_data;
}

toolchain::Value* EmitComparison(toolchain::CmpInst::Predicate predicate,
                            toolchain::Value* lhs_value, toolchain::Value* rhs_value,
                            toolchain::IRBuilderBase* b, absl::string_view name) {
  toolchain::Value* comparison_result;
  if (lhs_value->getType()->isIntegerTy()) {
    comparison_result =
        b->CreateICmp(predicate, lhs_value, rhs_value, AsStringRef(name));
  } else {
    comparison_result =
        b->CreateFCmp(predicate, lhs_value, rhs_value, AsStringRef(name));
  }
  // comparison_result is i1, but the NVPTX codegen incorrectly lowers i1
  // arrays. So we extend it to i8 so that it's addressable.
  return b->CreateZExt(comparison_result,
                       llvm_ir::PrimitiveTypeToIrType(PRED, b->getContext()));
}

// Internal helper that is called from emitted code to log an int64_t value with
// a tag.
static void LogS64(const char* tag, int64_t value) {
  LOG(INFO) << tag << " (int64_t): " << value;
}

void EmitLogging(const char* tag, toolchain::Value* value, toolchain::IRBuilderBase* b) {
  toolchain::FunctionType* log_function_type = toolchain::FunctionType::get(
      b->getVoidTy(), {b->getInt64Ty(), b->getInt64Ty()}, /*isVarArg=*/false);
  b->CreateCall(log_function_type,
                b->CreateIntToPtr(b->getInt64(absl::bit_cast<int64_t>(&LogS64)),
                                  b->getPtrTy()),
                {b->getInt64(absl::bit_cast<int64_t>(tag)), value});
}

void SetAlignmentMetadataForLoad(toolchain::LoadInst* load, uint64_t alignment) {
  toolchain::LLVMContext& context = load->getContext();
  toolchain::Type* int64_ty = toolchain::Type::getInt64Ty(context);
  toolchain::Constant* alignment_constant =
      toolchain::ConstantInt::get(int64_ty, alignment);
  toolchain::MDBuilder metadata_builder(context);
  auto* alignment_metadata =
      metadata_builder.createConstant(alignment_constant);
  load->setMetadata(toolchain::LLVMContext::MD_align,
                    toolchain::MDNode::get(context, alignment_metadata));
}

void SetDereferenceableMetadataForLoad(toolchain::LoadInst* load,
                                       uint64_t dereferenceable_bytes) {
  toolchain::LLVMContext& context = load->getContext();
  toolchain::Type* int64_ty = toolchain::Type::getInt64Ty(context);
  toolchain::Constant* dereferenceable_bytes_constant =
      toolchain::ConstantInt::get(int64_ty, dereferenceable_bytes);
  toolchain::MDBuilder metadata_builder(context);
  auto* dereferenceable_bytes_metadata =
      metadata_builder.createConstant(dereferenceable_bytes_constant);
  load->setMetadata(toolchain::LLVMContext::MD_dereferenceable,
                    toolchain::MDNode::get(context, dereferenceable_bytes_metadata));
}

toolchain::Instruction* AddRangeMetadata(int32_t lower, int32_t upper,
                                    toolchain::Instruction* inst,
                                    toolchain::Module* module) {
  if (toolchain::Triple(module->getTargetTriple()).isSPIR()) {
    return inst;
  }
  toolchain::LLVMContext& context = inst->getParent()->getContext();
  toolchain::IntegerType* i32 = toolchain::Type::getInt32Ty(context);
  inst->setMetadata(
      toolchain::LLVMContext::MD_range,
      toolchain::MDNode::get(
          context,
          {toolchain::ConstantAsMetadata::get(toolchain::ConstantInt::get(i32, lower)),
           toolchain::ConstantAsMetadata::get(toolchain::ConstantInt::get(i32, upper))}));
  return inst;
}

std::string IrName(absl::string_view a) {
  std::string s(a);
  s.erase(std::remove(s.begin(), s.end(), '%'), s.end());
  return s;
}

std::string IrName(absl::string_view a, absl::string_view b) {
  if (!a.empty() && !b.empty()) {
    return IrName(absl::StrCat(a, ".", b));
  }
  return IrName(absl::StrCat(a, b));
}

std::string IrName(const HloInstruction* a, absl::string_view b) {
  return IrName(a->name(), b);
}

mlir::OwningOpRef<mlir::ModuleOp> CreateMlirModuleOp(
    mlir::Location loc, std::optional<toolchain::StringRef> name) {
  return mlir::OwningOpRef<mlir::ModuleOp>(
      /*ALLOW_MLIR_MODULE_OP_CREATE*/ mlir::ModuleOp::create(std::move(loc),
                                                             std::move(name)));
}

std::string SanitizeFunctionName(std::string function_name) {
  // The backend with the strictest requirements on function names is NVPTX, so
  // we sanitize to its requirements.
  //
  // A slightly stricter version of the NVPTX requirements is that names match
  // /[a-zA-Z_$][a-zA-Z0-9_$]*/, with the exception that the names "_" and "$"
  // are illegal.

  // Sanitize chars in function_name.
  std::transform(function_name.begin(), function_name.end(),
                 function_name.begin(), [](char c) {
                   if (('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z') ||
                       ('0' <= c && c <= '9') || c == '_' || c == '$') {
                     return c;
                   }
                   return '_';
                 });

  // Ensure the name isn't empty.
  if (function_name.empty()) {
    function_name = "__unnamed";
  }

  // Ensure the name doesn't start with a number.
  if (!function_name.empty() && function_name[0] >= '0' &&
      function_name[0] <= '9') {
    function_name.insert(function_name.begin(), '_');
  }

  // Ensure the name isn't "_" or "$".
  if (function_name == "_" || function_name == "$") {
    function_name += '_';
  }

  return function_name;
}

void SetToFirstInsertPoint(toolchain::BasicBlock* blk,
                           toolchain::IRBuilderBase* builder) {
  builder->SetInsertPoint(blk, blk->getFirstInsertionPt());
}

void SetToLastInsertPoint(toolchain::BasicBlock* blk, toolchain::IRBuilderBase* builder) {
  if (toolchain::Instruction* terminator = blk->getTerminator()) {
    builder->SetInsertPoint(terminator);
  } else {
    builder->SetInsertPoint(blk);
  }
}

int64_t ByteSizeOf(const Shape& shape, const toolchain::DataLayout& data_layout) {
  unsigned pointer_size = data_layout.getPointerSize();
  return ShapeUtil::ByteSizeOf(shape, pointer_size);
}

toolchain::FastMathFlags GetCpuFastMathFlags(const HloModuleConfig& module_config) {
  toolchain::FastMathFlags flags;
  const auto& options = module_config.debug_options();
  if (!options.xla_cpu_enable_fast_math()) {
    return flags;
  }
  // Fast implies AllowReassoc, NoInfs, NoNaNs, NoSignedZeros, AllowReciprocal,
  // AllowContract, and ApproxFunc.
  flags.setFast();
  flags.setNoNaNs(!options.xla_cpu_fast_math_honor_nans());
  flags.setNoInfs(!options.xla_cpu_fast_math_honor_infs());
  flags.setAllowReciprocal(!options.xla_cpu_fast_math_honor_division());
  flags.setApproxFunc(!options.xla_cpu_fast_math_honor_functions());
  return flags;
}

std::map<int, toolchain::MDNode*> MergeMetadata(
    toolchain::LLVMContext* context, const std::map<int, toolchain::MDNode*>& a,
    const std::map<int, toolchain::MDNode*>& b) {
  // We should extend this as needed to deal with other kinds of metadata like
  // !dereferenceable and !range.

  std::map<int, toolchain::MDNode*> result;
  for (auto kind_md_pair : a) {
    if (kind_md_pair.first == toolchain::LLVMContext::MD_alias_scope) {
      toolchain::SmallVector<toolchain::Metadata*, 8> union_of_scopes;
      toolchain::SmallPtrSet<toolchain::Metadata*, 8> scope_set;
      for (const auto& scope_a : kind_md_pair.second->operands()) {
        scope_set.insert(toolchain::cast<toolchain::MDNode>(scope_a.get()));
        union_of_scopes.push_back(toolchain::cast<toolchain::MDNode>(scope_a.get()));
      }
      auto it = b.find(kind_md_pair.first);
      if (it != b.end()) {
        for (const auto& scope_b : it->second->operands()) {
          if (!scope_set.count(toolchain::cast<toolchain::MDNode>(scope_b.get()))) {
            union_of_scopes.push_back(toolchain::cast<toolchain::MDNode>(scope_b.get()));
          }
        }
      }
      result[toolchain::LLVMContext::MD_alias_scope] =
          toolchain::MDNode::get(*context, union_of_scopes);
    } else if (kind_md_pair.first == toolchain::LLVMContext::MD_noalias) {
      toolchain::SmallVector<toolchain::Metadata*, 8> intersection_of_scopes;
      toolchain::SmallPtrSet<toolchain::Metadata*, 8> scope_set;
      for (const auto& scope_a : kind_md_pair.second->operands()) {
        scope_set.insert(toolchain::cast<toolchain::MDNode>(scope_a.get()));
      }
      auto it = b.find(kind_md_pair.first);
      if (it != b.end()) {
        for (const auto& scope_b : it->second->operands()) {
          if (scope_set.count(toolchain::cast<toolchain::MDNode>(scope_b))) {
            intersection_of_scopes.push_back(toolchain::cast<toolchain::MDNode>(scope_b));
          }
        }
      }
      if (!intersection_of_scopes.empty()) {
        result[toolchain::LLVMContext::MD_noalias] =
            toolchain::MDNode::get(*context, intersection_of_scopes);
      }
    }
  }
  return result;
}

void DumpIrIfEnabled(const HloModule& hlo_module,
                     const toolchain::Module& llvm_module, bool optimized,
                     absl::string_view filename_suffix) {
  if (!DumpingEnabledForHloModule(hlo_module)) {
    return;
  }
  tsl::profiler::ScopedAnnotation annotation([&] {
    return absl::StrFormat("XlaDumpLlvmIr:#module=%s,program_id=%d#",
                           hlo_module.name(), hlo_module.unique_id());
  });
  // We can end up compiling different modules with the same name when using
  // XlaJitCompiledCpuFunction::Compile.  Avoid overwriting IR files previously
  // dumped from the same process in such cases.
  std::string suffix =
      absl::StrCat(filename_suffix, filename_suffix.empty() ? "" : ".", "ir-",
                   optimized ? "with" : "no", "-opt");
  DumpToFileInDirOrStdout(hlo_module, "", absl::StrCat(suffix, ".ll"),
                          DumpToString(&llvm_module));
}

toolchain::Function* CreateCpuFunction(toolchain::FunctionType* function_type,
                                  toolchain::GlobalValue::LinkageTypes linkage,
                                  const HloModuleConfig& module_config,
                                  absl::string_view name,
                                  toolchain::Module* module) {
  toolchain::Function* function =
      toolchain::Function::Create(function_type, linkage, AsStringRef(name), module);
  function->setCallingConv(toolchain::CallingConv::C);
  function->addFnAttr("no-frame-pointer-elim", "false");

  // Generate unwind information so that GDB can crawl through the stack frames
  // created by the JIT compiled code.
  function->setUWTableKind(toolchain::UWTableKind::Default);

  // Tensorflow always flushes denormals to zero, let LLVM know that flushing
  // denormals is safe. This allows vectorization using ARM's neon instruction
  // set.
  function->addFnAttr("denormal-fp-math", "preserve-sign");

  // Add the optimize attribute to the function if optimizing for size. This
  // controls internal behavior of some optimization passes (e.g. loop
  // unrolling).
  if (cpu::options::OptimizeForSizeRequested(module_config)) {
    function->addFnAttr(toolchain::Attribute::OptimizeForSize);
  }

  return function;
}

unsigned GetGlobalMemoryAddressSpace() { return 1; }

toolchain::GlobalVariable* GetOrCreateVariableForRngState(toolchain::Module* module,
                                                     toolchain::IRBuilderBase* b) {
  static const char* kRngStateVariableName = "rng_state";
  toolchain::GlobalVariable* state_ptr =
      module->getNamedGlobal(kRngStateVariableName);
  if (!state_ptr) {
    toolchain::Type* state_type = b->getInt128Ty();
    // Use a non-zero initial value as zero state can cause the result of the
    // first random number generation not passing the chi-square test. The
    // values used here are arbitrarily chosen, any non-zero values should be
    // fine.
    state_ptr = new toolchain::GlobalVariable(
        /*M=*/*module,
        /*Ty=*/state_type,
        /*isConstant=*/false,
        /*Linkage=*/toolchain::GlobalValue::PrivateLinkage,
        /*Initializer=*/toolchain::ConstantInt::get(b->getInt128Ty(), 0x7012395ull),
        /*Name=*/kRngStateVariableName,
        /*InsertBefore=*/nullptr,
        /*TLMode=*/toolchain::GlobalValue::NotThreadLocal,
        /*AddressSpace=*/GetGlobalMemoryAddressSpace(),
        /*isExternallyInitialized=*/false);
  }
  return state_ptr;
}

toolchain::Value* RngGetAndUpdateState(uint64_t delta, toolchain::Module* module,
                                  toolchain::IRBuilderBase* builder) {
  toolchain::GlobalVariable* state_ptr =
      GetOrCreateVariableForRngState(module, builder);
  toolchain::LoadInst* state_value_old =
      builder->CreateLoad(state_ptr->getValueType(), state_ptr, "load_state");
  toolchain::Value* state_value_new = builder->CreateAdd(
      state_value_old,
      toolchain::ConstantInt::get(state_value_old->getType(), delta));
  builder->CreateStore(state_value_new, state_ptr);
  return state_value_old;
}

toolchain::BasicBlock* EmitReturnBlock(toolchain::IRBuilderBase* b) {
  toolchain::Function* function = b->GetInsertBlock()->getParent();
  toolchain::Module* module = b->GetInsertBlock()->getModule();
  toolchain::IRBuilderBase::InsertPointGuard guard(*b);
  toolchain::BasicBlock* early_return =
      toolchain::BasicBlock::Create(/*Context=*/module->getContext(),
                               /*Name=*/"early_return",
                               /*Parent=*/function);
  b->SetInsertPoint(early_return);
  b->CreateRetVoid();
  return early_return;
}

void EmitEarlyReturn(toolchain::Value* condition, toolchain::IRBuilderBase* b,
                     toolchain::BasicBlock* return_block) {
  if (!return_block) {
    return_block = EmitReturnBlock(b);
  }

  toolchain::BasicBlock* continued;

  // Implicitly check whtether we are already at the end of unterminated block.
  if (b->GetInsertBlock()->getTerminator() == nullptr) {
    // If we are generating code into an incomplete basic block we can just
    // create a new basic block to jump to after our conditional branch.
    continued = llvm_ir::CreateBasicBlock(/*insert_before=*/nullptr,
                                          /*name=*/"", b);
  } else {
    // If we are generating code into a basic block that already has code, we
    // need to split that block so as to not disturb the existing code.
    auto original = b->GetInsertBlock();
    continued = original->splitBasicBlock(b->GetInsertPoint());
    // Remove the auto-generated unconditional branch to replace with our
    // conditional branch.
    original->getTerminator()->eraseFromParent();
    b->SetInsertPoint(original);
  }

  b->CreateCondBr(condition, continued, return_block);
  b->SetInsertPoint(continued, continued->getFirstInsertionPt());
}

}  // namespace llvm_ir
}  // namespace xla
