/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, August 10, 2025.
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

#include "machina/compiler/aot/embedded_constant_buffers.h"

#include <sys/types.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "toolchain/ADT/ArrayRef.h"
#include "toolchain/ADT/SmallVector.h"
#include "toolchain/IR/Constants.h"
#include "toolchain/IR/GlobalVariable.h"
#include "toolchain/IR/LLVMContext.h"
#include "toolchain/IR/LegacyPassManager.h"
#include "toolchain/IR/Module.h"
#include "toolchain/MC/TargetRegistry.h"
#include "toolchain/Support/Alignment.h"
#include "toolchain/Support/CodeGen.h"
#include "toolchain/Support/raw_ostream.h"
#include "toolchain/Target/TargetMachine.h"
#include "toolchain/Target/TargetOptions.h"
#include "toolchain/TargetParser/Triple.h"
#include "machina/xla/backends/cpu/alignment.h"
#include "machina/xla/service/llvm_ir/llvm_type_conversion_util.h"
#include "machina/xla/tsl/platform/errors.h"
#include "machina/xla/tsl/platform/statusor.h"
#include "machina/xla/util.h"

namespace machina {
namespace tfcompile {

using xla::llvm_ir::AsStringRef;

void ConstantToEmbed::SerializeIntoBuffer(absl::Span<const uint8_t> buffer) {
  // The header has to be padded to 64 bytes so that the pointer to the
  // constant is always 64-byte aligned.
  const size_t header_size = xla::cpu::Align();
  const size_t padding_size = header_size - sizeof(uint64_t);

  const uint64_t buffer_size = buffer.size();
  data_buffer.resize(header_size + buffer_size);

  std::memcpy(data_buffer.data(), &buffer_size, sizeof(uint64_t));
  std::memset(data_buffer.data() + sizeof(uint64_t), 0, padding_size);
  std::memcpy(data_buffer.data() + header_size, buffer.data(), buffer.size());
}

static absl::Status AddBufferToLlvmModule(
    toolchain::Module* module, const ConstantToEmbed& constant_to_embed,
    absl::string_view unique_identifier,
    std::string& constant_array_symbol_name) {
  if (constant_to_embed.data().empty()) {
    return xla::Internal(
        "Constant buffer shouldn't be empty, it should at least contain the "
        "size of the buffer.");
  }

  absl::Span<const uint8_t> buffer_contents = constant_to_embed.data();

  toolchain::Constant* buffer_initializer = toolchain::ConstantDataVector::get(
      module->getContext(),
      toolchain::ArrayRef<uint8_t>(buffer_contents.data(), buffer_contents.size()));

  constant_array_symbol_name =
      absl::StrCat(unique_identifier, "_constant_buffer_contents");
  toolchain::GlobalVariable* global_variable = new toolchain::GlobalVariable(
      *module, buffer_initializer->getType(),
      /*isConstant=*/true, toolchain::GlobalValue::ExternalLinkage,
      buffer_initializer, AsStringRef(constant_array_symbol_name));

  global_variable->setAlignment(toolchain::Align(xla::cpu::Align()));

  return absl::OkStatus();
}

static absl::StatusOr<std::string> CodegenModule(
    toolchain::TargetMachine* target_machine, std::unique_ptr<toolchain::Module> module) {
  toolchain::SmallVector<char, 0> stream_buffer;
  toolchain::raw_svector_ostream ostream(stream_buffer);
  toolchain::legacy::PassManager codegen_passes;

  if (target_machine->addPassesToEmitFile(codegen_passes, ostream, nullptr,
                                          toolchain::CodeGenFileType::ObjectFile)) {
    return xla::Internal(
        "Could not create pass pipeline to generate object file");
  }

  codegen_passes.run(*module);

  return std::string(stream_buffer.begin(), stream_buffer.end());
}

static absl::StatusOr<std::unique_ptr<toolchain::TargetMachine>>
GetTargetMachineFromTriple(absl::string_view target_triple) {
  std::string error;
  std::string normalized_triple =
      toolchain::Triple::normalize(AsStringRef(absl::string_view(target_triple)));
  const toolchain::Target* target =
      toolchain::TargetRegistry::lookupTarget(normalized_triple, error);
  if (target == nullptr) {
    return xla::Internal("TargetRegistry::lookupTarget failed: %s",
                         error.c_str());
  }

  return absl::WrapUnique(target->createTargetMachine(
      normalized_triple, /*CPU=*/"",
      /*Features=*/"", toolchain::TargetOptions(), std::nullopt));
}

absl::StatusOr<EmbeddedConstantBuffers> CreateEmbeddedConstantBuffers(
    absl::string_view target_triple,
    absl::Span<ConstantToEmbed> constants_to_embed) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<toolchain::TargetMachine> target_machine,
                      GetTargetMachineFromTriple(target_triple));

  toolchain::LLVMContext llvm_context;
  auto module_with_serialized_proto = std::make_unique<toolchain::Module>(
      "embedded_constant_data_module", llvm_context);

  EmbeddedConstantBuffers result;

  for (const ConstantToEmbed& constant_to_embed : constants_to_embed) {
    std::string constant_array_symbol_name;

    TF_RETURN_IF_ERROR(AddBufferToLlvmModule(
        module_with_serialized_proto.get(), constant_to_embed,
        constant_to_embed.symbol_prefix, constant_array_symbol_name));

    // NOTE: Some targets will prepend an underscore to the symbol name at
    // compilation time. Using asm allows us to ensure the given symbol name is
    // always used. https://clang.toolchain.org/docs/AttributeReference.html#asm
    std::string cpp_variable_decl =
        absl::StrCat("extern \"C\" char ", constant_array_symbol_name,
                     "[] asm(\"", constant_array_symbol_name, "\");");

    // NOTE: The actual constant is located after the header which consists of
    // an 8 bit size and padding to 64 bytes so that the pointer to the constant
    // is always 64-byte aligned.
    std::string cpp_access_shim = absl::StrFormat(
        R"(
    [](char* buffer) -> std::pair<uint64_t, char*> {
      uint64_t buffer_size;
      std::memcpy(&buffer_size, buffer, sizeof(uint64_t));
      return {buffer_size, buffer + %d};
    }(%s)
    )",
        xla::cpu::Align(), constant_array_symbol_name);
    result.variable_decls.push_back(
        {constant_array_symbol_name, cpp_variable_decl, cpp_access_shim});
  }

  TF_ASSIGN_OR_RETURN(result.object_file_data,
                      CodegenModule(target_machine.get(),
                                    std::move(module_with_serialized_proto)));
  return result;
}

}  // namespace tfcompile
}  // namespace machina
