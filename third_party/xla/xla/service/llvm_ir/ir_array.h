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

#ifndef MACHINA_MACHINA_XLA_SERVICE_LLVM_IR_IR_ARRAY_H_
#define MACHINA_MACHINA_XLA_SERVICE_LLVM_IR_IR_ARRAY_H_

#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "toolchain/ADT/STLExtras.h"
#include "toolchain/IR/Constants.h"
#include "toolchain/IR/IRBuilder.h"
#include "toolchain/IR/Value.h"
#include "machina/xla/layout.h"
#include "machina/xla/map_util.h"
#include "machina/xla/shape.h"
#include "machina/xla/shape_util.h"
#include "machina/xla/tsl/platform/logging.h"
#include "machina/xla/types.h"
#include "machina/xla/xla_data.pb.h"

namespace xla {
namespace llvm_ir {

// IrArray represents an XLA array at the LLVM IR level. This class
// encapsulates a base pointer to the buffer holding the array (as an LLVM
// Value) and the shape of the array. The class includes methods for emitting
// LLVM IR sequences which access elements of the array at a multidimensional
// index (eg, [x, y, z] in a 3-dimensional array). Arbitrary shape and layouts
// are supported.
class IrArray {
 public:
  // A multidimensional index into an IrArray. The order of the runtime indices
  // (multidim) corresponds to the order of dimensions in the Shape passed to
  // the constructor.
  //
  // This may also keep a linear index and the layout and dimensions it was
  // emitted for; if the shape where this `Index` is used matches, the linear
  // index may be used, potentially sparing the cost of computing the
  // multidimensional index, which LLVM DCE can delete.
  class Index {
   public:
    // Constructs an index for a scalar shape.
    explicit Index(toolchain::Type* index_ty) : index_type_(index_ty) {
      CHECK(index_ty->isIntegerTy());
    }

    // Constructs an index from linear index "linear" and computes the
    // multi-dimensional index from "linear" and "shape". "b" is the IR
    // builder to emit the index of each dimension in the multi-dimensional
    // index.
    //
    // Precondition: "shape" has a layout.
    Index(toolchain::Value* linear, const Shape& shape, toolchain::IRBuilderBase* b);

    // As before, but also take a multidim to reuse.  multidim.size()
    // == shape.dimensions_size() must be true.  If some of the multidim element
    // are null we will use the value that would be used if
    // deliearized from linear.
    Index(toolchain::Value* linear, absl::Span<toolchain::Value* const> multidim,
          const Shape& shape, toolchain::IRBuilderBase* b);

    // Similar to the above constructor except using "dynamic_dims" instead of
    // shape's static dimension to constructs the index.
    Index(toolchain::Value* linear, const Shape& shape,
          absl::Span<toolchain::Value*> dynamic_dims, toolchain::IRBuilderBase* b);

    // Constructs an index from a multi-dimensional index. 'shape' is the shape
    // for which the multi-dimensional index is used. 'index_type' is the type
    // of the index.
    //
    // Precondition: "shape" has a layout.
    Index(absl::Span<toolchain::Value* const> multidim, const Shape& shape,
          toolchain::Type* index_type);

    // Same as above, but only the dimensions of the shape without layout is
    // passed. The layout is assumed to be the default (descending
    // minor-to-major) layout.
    Index(absl::Span<toolchain::Value* const> multidim,
          absl::Span<int64_t const> dimensions, toolchain::Type* index_type);

    // Returns an index that adds `addend` to the given `dim` of the object.
    Index AddOffsetToDim(toolchain::Value* addend, int64_t dim,
                         toolchain::IRBuilderBase* b) const {
      Index with_offset = *this;
      with_offset.linear_ = nullptr;
      with_offset.multidim_[dim] =
          b->CreateAdd(with_offset.multidim_[dim], addend);
      return with_offset;
    }

    Index AddOffset(absl::Span<toolchain::Value* const> offsets,
                    toolchain::IRBuilderBase* b) const {
      CHECK_EQ(multidim_.size(), offsets.size());
      Index with_offset = *this;
      with_offset.linear_ = nullptr;
      for (auto&& [dim, offset] : toolchain::zip(with_offset.multidim_, offsets)) {
        dim = b->CreateAdd(dim, offset);
      }
      return with_offset;
    }

    const std::vector<toolchain::Value*>& multidim() const { return multidim_; }
    const std::vector<int64_t>& dims() const { return dims_; }
    toolchain::Value* linear() const { return linear_; }

    size_t size() const { return multidim().size(); }

    toolchain::Value* operator[](size_t i) const { return multidim()[i]; }

    using const_iterator = std::vector<toolchain::Value*>::const_iterator;

    const_iterator begin() const { return multidim().begin(); }
    const_iterator end() const { return multidim().end(); }

    bool LinearValidOnShape(const Shape& a) const;

    static bool ShapeIsCompatible(const Shape& a, const Shape& b);

    bool ShapeIsCompatible(const Shape& a) const {
      return ShapeIsCompatible(a, AsShapeWithType(a.element_type()));
    }

    Shape AsShapeWithType(PrimitiveType element_type) const {
      return ShapeUtil::MakeShapeWithDenseLayout(element_type, dims_,
                                                 layout_.minor_to_major());
    }

    // Given that "this" is the target index of a reshape from `input_shape`
    // to `output_shape`, returns the source index.
    Index SourceIndexOfReshape(const Shape& output_shape,
                               const Shape& input_shape,
                               toolchain::IRBuilderBase* builder) const;

    // Returns the index into the source operand from which a slice operation
    // selects a value to be placed into index "this". The slice is described
    // by starting indices `starts` and stride values `strides`.
    //
    // Precondition: "this" is an index into a slice whose operand shape is
    // `operand_shape`.
    Index SourceIndexOfSlice(const Shape& operand_shape,
                             absl::Span<const int64_t> starts,
                             absl::Span<const int64_t> strides,
                             toolchain::IRBuilderBase* builder) const;

    // Given that "this" is the target index of a transpose from `operand_shape`
    // to `shape` with the given dimension mapping, returns the source index.
    Index SourceIndexOfTranspose(
        const Shape& shape, const Shape& operand_shape,
        absl::Span<const int64_t> dimension_mapping) const;

    // Given that "this" is the target index of a bitcast from `operand_shape`
    // to `shape`, returns the source index.
    Index SourceIndexOfBitcast(const Shape& shape, const Shape& operand_shape,
                               toolchain::IRBuilderBase* builder) const;
    // Same as above, but for bitcasts from `operand_shape` to `this->dims`.
    Index SourceIndexOfBitcast(const Shape& operand_shape,
                               toolchain::IRBuilderBase* builder) const;

    // Given that "this" is the target index of a broadcast from `operand_shape`
    // to `shape` with the given dimension mapping, returns the source index.
    Index SourceIndexOfBroadcast(const Shape& shape, const Shape& operand_shape,
                                 absl::Span<const int64_t> dimension_mapping,
                                 toolchain::IRBuilderBase* builder) const;

    // Linearizes the index into the given shape, i.e. reshapes it to rank-1 and
    // returns the index into the sole dimension 0 of the new shape.
    toolchain::Value* Linearize(absl::Span<const int64_t> dimensions,
                           toolchain::IRBuilderBase* builder) const;

    // Linearizes the index into the given dynamic dimensions.
    toolchain::Value* Linearize(const std::vector<toolchain::Value*>& dynamic_dims,
                           toolchain::IRBuilderBase* builder) const;

    toolchain::Type* GetType() const { return index_type_; }

    toolchain::Constant* GetConstantWithIndexType(int64_t c) const {
      // The LLVM function makes sure that the value can be represented by the
      // specified type, see ConstantInt::ConstantInt(IntegerType *Ty, const
      // APInt &V).
      return toolchain::ConstantInt::get(index_type_, c);
    }

   private:
    // Constructs an index from both a multi-dimensional index and a linear
    // index. 'shape' is the shape on which the index is used. 'index_type' is
    // the type of the index.
    //
    // Precondition: "shape" has a layout.
    Index(absl::Span<toolchain::Value* const> multidim, toolchain::Value* linear,
          const Shape& shape, toolchain::Type* index_type);

    void Delinearize(std::vector<toolchain::Value*>* multidim, toolchain::Value* linear,
                     const Shape& shape, toolchain::IRBuilderBase* b) const;

    // Delinearize the linear index with the dynamic dimensions.
    void Delinearize(std::vector<toolchain::Value*>* multidim, toolchain::Value* linear,
                     const Shape& shape, absl::Span<toolchain::Value*> dynamic_dims,
                     toolchain::IRBuilderBase* b) const;

    std::vector<toolchain::Value*> multidim_;

    // These values are purely for efficiency; `multidim_` is enough to find the
    // element at a given `Index`, but if a loop is emitted with a linear index
    // space, that linear index can be saved in `linear_`, and the layout and
    // dimensions of the shape the loop was emitted for in `layout_` and
    // `dims_`, and if the `Index` is used in another array, and its layout and
    // dimensions match, the linear index can be used, sparing the cost of
    // computing `multidim_`, which LLVM DCE could potentially so delete.
    // Modifying `multidim_` after construction nullifies `linear_`, lest it
    // be used wrongly, as it would be valid no more.
    // If a loop is emitted with a multidimensional index space, `linear_` would
    // be null and `layout_` and `dims_` would be ignored.
    toolchain::Value* linear_ = nullptr;
    Layout layout_;
    std::vector<int64_t> dims_;

    toolchain::Type* index_type_;
  };

  // Default constructor. Constructs an IrArray in a null status.
  IrArray() : base_ptr_(nullptr) {}

  // Construct an IrArray with the given base pointer, pointee type, and shape.
  // base_ptr is a pointer type pointing to the first element(lowest address)
  // of the array.
  //
  // For packed arrays, base_ptr points to packed memory with the correct number
  // of elements when unpacked. pointee_type should be an iN array in this case,
  // and reads and writes will return or take in iN values. IrArray internally
  // reads or writes i8 values, by treating base_ptr as an i8 array and
  // masking/shifting on the fly. IrArray does not directly read/write iN
  // values, since arrays of iN values in LLVM are not packed (every element of
  // an LLVM IR array must have unique address).
  IrArray(toolchain::Value* base_ptr, toolchain::Type* pointee_type, Shape shape);

  // Default implementations of copying and moving.
  IrArray(IrArray&& other) noexcept = default;
  IrArray(const IrArray& other) = default;
  IrArray& operator=(IrArray&& other) noexcept = default;
  IrArray& operator=(const IrArray& other) = default;

  toolchain::Value* GetBasePointer() const { return base_ptr_; }
  toolchain::Type* GetBasePointeeType() const { return pointee_type_; }
  toolchain::Type* GetElementLlvmType() const { return element_type_; }

  const Shape& GetShape() const { return shape_; }

  // Emit a sequence of instructions to compute the address of the element in
  // the given array at the given index. Returns the address of the element as
  // an LLVM Value.
  //
  // The optional name is useful for debugging when looking at
  // the emitted LLVM IR.
  //
  // `bit_offset` contains the offset of the element inside the address.
  toolchain::Value* EmitArrayElementAddress(
      const Index& index, toolchain::IRBuilderBase* b, absl::string_view name = "",
      bool use_linear_index = true, toolchain::Value** bit_offset = nullptr) const;

  // Attach metadata this IrArray instance knows about to "instruction".
  void AnnotateLoadStoreInstructionWithMetadata(
      toolchain::Instruction* instruction) const;

  // Emit IR to read an array element at the given index. Returns the read
  // result (effectively, a Value loaded from memory). This method seamlessly
  // handles scalar shapes by broadcasting their value to all indices (index is
  // ignored).
  //
  // The optional name is useful for debugging when looking at
  // the emitted LLVM IR.
  // 'use_linear_index' can be used to specify whether the linear index (if
  // available) or the multi-dimensional index should be used.
  toolchain::Value* EmitReadArrayElement(const Index& index, toolchain::IRBuilderBase* b,
                                    absl::string_view name = "",
                                    bool use_linear_index = true) const;

  // Emit IR to write the given value to the array element at the given index.
  // 'use_linear_index' can be used to specify whether the linear index (if
  // available) or the multi-dimensional index should be used.
  //
  // For packed arrays, only part of the byte in the array is written. First
  // the appropriate byte is read from the array, then a subset of bits are
  // modified and written back. To avoid race conditions, the caller must ensure
  // that the different values within a byte are not written to in parallel.
  void EmitWriteArrayElement(const Index& index, toolchain::Value* value,
                             toolchain::IRBuilderBase* b,
                             bool use_linear_index = true) const;

  // Returns a new IrArray whose shape is "new_shape" and base pointer is a
  // bitcast of the base pointer of "this" IrArray.
  // 'use_linear_index' can be used to specify whether the linear index (if
  // available) or the multi-dimensional index should be used.
  IrArray CastToShape(const Shape& new_shape, toolchain::IRBuilderBase* b) const;

  void AddAliasScopeMetadata(toolchain::MDNode* alias_scope) {
    CHECK_NE(alias_scope, nullptr);
    AddMetadata(toolchain::LLVMContext::MD_alias_scope, alias_scope);
  }

  void AddNoaliasMetadata(toolchain::MDNode* noalias) {
    CHECK_NE(noalias, nullptr);
    AddMetadata(toolchain::LLVMContext::MD_noalias, noalias);
  }

  // Promises LLVM that the data pointed to by this IrArray never changes after
  // it's first loaded.
  //
  // The temporal scope of this promise is the "whole program" from LLVM's point
  // of view, but how this translates to HLOs differs between backends.
  //
  // In the single-threaded CPU backend, we emit one function that
  // runs all the HLOs in sequence, so the whole program is the whole HLO
  // module.
  //
  // In the GPU backend, we emit one GPU kernel per top-level HLO (i.e. per HLO
  // in the entry computation).  From LLVM's perspective, launching a new kernel
  // is like launching a new program, and so the whole program is one top-level
  // HLO.  Since the scope of the promise is smaller than in the CPU backend, we
  // can mark more things as invariant in the GPU backend.
  //
  // Marking loads as invariant is particularly helpful on GPUs because
  // invariant loads can be lowered to PTX ld.global.nc (equivalent to CUDA's
  // __ldg intrinsic).  These loads use a special cache, and can be
  // significantly faster than regular loads.
  void MarkInvariantOverWholeProgram(toolchain::LLVMContext* context) {
    if (is_invariant_) {
      return;
    }
    is_invariant_ = true;
    AddMetadata(toolchain::LLVMContext::MD_invariant_load,
                toolchain::MDNode::get(*context, {}));
  }

  const std::map<int, toolchain::MDNode*>& metadata() const { return metadata_; }

 private:
  // Add the specified LLVM IR metadata to loads/stores associated with this
  // IrArray.
  void AddMetadata(int kind, toolchain::MDNode* md) {
    InsertOrDie(&metadata_, kind, md);
  }

  // Like EmitArrayElementAddress, but always uses a linear index.
  toolchain::Value* EmitLinearArrayElementAddress(
      const Index& index, toolchain::IRBuilderBase* b, absl::string_view name = "",
      toolchain::Value** bit_offset = nullptr) const;

  // Address of the base of the array as an LLVM Value.
  toolchain::Value* base_ptr_;

  // The pointee type of base_ptr_;
  toolchain::Type* pointee_type_;

  // The LLVM type of the elements in the array.
  toolchain::Type* element_type_;

  // Shape of the XLA array.
  Shape shape_;

  // The list of key/value pairs used when attaching metadata to emitted
  // loads/stores for this array.  They keys are the metadata kinds and the
  // values are the metadata nodes.
  std::map<int, toolchain::MDNode*> metadata_;

  bool is_invariant_ = false;
};

}  // namespace llvm_ir
}  // namespace xla

#endif  // MACHINA_MACHINA_XLA_SERVICE_LLVM_IR_IR_ARRAY_H_
