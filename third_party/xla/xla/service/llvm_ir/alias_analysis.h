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

#ifndef MACHINA_MACHINA_XLA_SERVICE_LLVM_IR_ALIAS_ANALYSIS_H_
#define MACHINA_MACHINA_XLA_SERVICE_LLVM_IR_ALIAS_ANALYSIS_H_

#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "toolchain/IR/Metadata.h"
#include "toolchain/IR/Module.h"
#include "machina/xla/hlo/ir/hlo_instruction.h"
#include "machina/xla/service/buffer_assignment.h"
#include "machina/xla/service/llvm_ir/ir_array.h"
#include "machina/xla/shape_util.h"
#include "machina/xla/types.h"

namespace xla {
namespace llvm_ir {

// Helper functionality used to augment the LLVM IR emitted with alias-scope
// metadata.
class AliasAnalysis {
 public:
  AliasAnalysis(const HloModule& module, const BufferAssignment& assignment,
                toolchain::LLVMContext* context)
      : module_(module), assignment_(assignment), context_(context) {}

  // Augments IrArray with aliasing information.
  void AddAliasingInformationToIrArray(const HloInstruction& hlo,
                                       llvm_ir::IrArray* array,
                                       const ShapeIndex& index = {});

 private:
  // Returns a unique alias domain for this emitter.
  toolchain::MDNode* GetAliasDomain();

  // Returns an alias.scope metadata node corresponding to a given buffer slice.
  toolchain::MDNode* GetAliasScopeMetadataForBuffer(
      const BufferAllocation::Slice& buffer_slice, toolchain::MDNode* domain);

  // Returns a noalias metadata node corresponding to a given buffer slice.
  //
  // |buffer_slice| is the buffer slice.
  //
  // |domain| corresponds to the alias scope domain as documented at
  // http://toolchain.org/docs/LangRef.html#noalias-and-alias-scope-metadata
  //
  // |hlo| is the instruction we are computing a noalias set for.
  toolchain::MDNode* GetNoaliasMetadataForBuffer(
      const BufferAllocation::Slice& buffer_slice, toolchain::MDNode* domain,
      const BufferAssignment& assignment, const HloInstruction& hlo);

  // The HLO module we are compiling for.
  const HloModule& module_;

  // Assignment of the temporary buffers needed by the computation and their
  // shape information.
  const BufferAssignment& assignment_;

  // The LLVM context which we are using for IR emission.
  toolchain::LLVMContext* context_;

  // Holds the alias domain for this computation.
  toolchain::MDNode* alias_domain_ = nullptr;

  // A map from a buffer slice to metadata corresponding to its alias.scope
  // metadata.  The index kParameterAliasSet is used to hold aliasing
  // information for parameters.
  absl::flat_hash_map<BufferAllocation::Slice, toolchain::MDNode*>
      alias_scope_metadata_;

  // A map from a buffer slice and producer to metadata corresponding to its
  // noalias metadata.
  absl::flat_hash_map<std::pair<BufferAllocation::Slice, const HloInstruction*>,
                      toolchain::MDNode*>
      noalias_metadata_;
};

}  // namespace llvm_ir
}  // namespace xla

#endif  // MACHINA_MACHINA_XLA_SERVICE_LLVM_IR_ALIAS_ANALYSIS_H_
