/* Copyright 2025 The OpenXLA Authors.

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

#include "machina/xla/codegen/intrinsic/simple_jit_runner.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/call_once.h"
#include "absl/log/log.h"
#include "toolchain/ADT/StringMap.h"
#include "toolchain/ExecutionEngine/JITEventListener.h"
#include "toolchain/ExecutionEngine/Orc/CompileUtils.h"
#include "toolchain/ExecutionEngine/Orc/Core.h"
#include "toolchain/ExecutionEngine/Orc/IRCompileLayer.h"
#include "toolchain/ExecutionEngine/Orc/LLJIT.h"
#include "toolchain/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "toolchain/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "toolchain/ExecutionEngine/SectionMemoryManager.h"
#include "toolchain/IR/BasicBlock.h"
#include "toolchain/IR/DerivedTypes.h"
#include "toolchain/IR/Function.h"
#include "toolchain/IR/IRBuilder.h"
#include "toolchain/IR/Instructions.h"
#include "toolchain/IR/LLVMContext.h"
#include "toolchain/IR/Module.h"
#include "toolchain/IR/Type.h"
#include "toolchain/IR/Value.h"
#include "toolchain/IR/Verifier.h"
#include "toolchain/MC/TargetRegistry.h"
#include "toolchain/Support/Alignment.h"
#include "toolchain/Support/Errc.h"
#include "toolchain/Support/Error.h"
#include "toolchain/Support/ErrorHandling.h"
#include "toolchain/Support/TargetSelect.h"
#include "toolchain/Support/TypeSize.h"
#include "toolchain/Support/raw_ostream.h"
#include "toolchain/Target/TargetMachine.h"
#include "toolchain/TargetParser/Host.h"
#include "toolchain/TargetParser/Triple.h"
#include "machina/xla/service/llvm_ir/llvm_util.h"
#include "machina/xla/xla_data.pb.h"

namespace xla::codegen::intrinsic {

namespace {
void initializeNativeTargets() {
  static absl::once_flag once;
  absl::call_once(once, []() {
    toolchain::InitializeNativeTarget();
    toolchain::InitializeNativeTargetAsmPrinter();
    toolchain::InitializeNativeTargetAsmParser();
  });
}
}  // namespace

JitRunner::JitRunner(std::unique_ptr<toolchain::Module> module,
                     std::unique_ptr<toolchain::LLVMContext> context) {
  initializeNativeTargets();
  tsc_ = std::make_unique<toolchain::orc::ThreadSafeContext>(std::move(context));
  perf_listener_ = toolchain::JITEventListener::createPerfJITEventListener();
  auto jit_builder = toolchain::orc::LLJITBuilder();
  if (perf_listener_ != nullptr) {
    jit_builder = std::move(jit_builder.setObjectLinkingLayerCreator(
        [&](toolchain::orc::ExecutionSession& ES) {
          auto obj_layer =
              std::make_unique<toolchain::orc::RTDyldObjectLinkingLayer>(
                  ES, [](const toolchain::MemoryBuffer& _) {
                    return std::make_unique<toolchain::SectionMemoryManager>();
                  });
          obj_layer->registerJITEventListener(*perf_listener_);
          return obj_layer;
        }));
  }
  auto jit_or_err = jit_builder.create();
  if (!jit_or_err) {
    toolchain::report_fatal_error(
        toolchain::Twine(toolchain::toString(jit_or_err.takeError())));
  }
  jit_ = std::move(jit_or_err.get());
  toolchain::orc::ThreadSafeModule tsm(std::move(module), *tsc_);
  toolchain::ExitOnError exit_on_err;
  exit_on_err(jit_->addIRModule(std::move(tsm)));
}

// Returns a JITed function that loops over a vectorized function.
// The original function is expected to have a signature like:
//   <VectorSize x RetType> func(<VectorSize x Arg1Type>, <VectorSize x
//   Arg2Type>, ...)
// This function creates a wrapper that bridges the gap between C++ array
// types and LLVM vector types. The returned std::function has a signature like:
// void fn(std:::array<ArgTypes, VectorSize>& return_array,
//   size_t iteration_count, size_t source_array_length, const
//   std::array<ArgTypes, VectorSize>&...)
toolchain::Expected<void*> JitRunner::CreateVectorWrapperWithLoop(
    const std::string& original_function_name, size_t vector_size,
    PrimitiveType ret_type, std::vector<PrimitiveType> arg_types) {
  return tsc_->withContextDo([&](toolchain::LLVMContext* ctx)
                                 -> toolchain::Expected<void*> {
    auto wrapper_module_owner = std::make_unique<toolchain::Module>(
        original_function_name + "_wrapper_module", *ctx);
    toolchain::Module* wrapper_module = wrapper_module_owner.get();
    toolchain::IRBuilder<> builder(*ctx);

    auto vec_type = [&](PrimitiveType type) {
      return toolchain::VectorType::get(llvm_ir::PrimitiveTypeToIrType(type, *ctx),
                                   toolchain::ElementCount::getFixed(vector_size));
    };

    toolchain::Type* ret_vec_type = vec_type(ret_type);
    std::vector<toolchain::Type*> arg_vec_types;
    arg_vec_types.reserve(arg_types.size());
    for (PrimitiveType arg_type : arg_types) {
      arg_vec_types.push_back(vec_type(arg_type));
    }

    toolchain::FunctionType* original_func_type =
        toolchain::FunctionType::get(ret_vec_type, arg_vec_types, false);
    toolchain::Function* original_func = toolchain::Function::Create(
        original_func_type, toolchain::Function::ExternalLinkage,
        original_function_name, *wrapper_module);

    std::vector<toolchain::Type*> wrapper_arg_types;
    // 1. Pointer to write the return data
    wrapper_arg_types.push_back(ret_vec_type->getScalarType()->getPointerTo());
    // 2. Iteration count, passed by value
    wrapper_arg_types.push_back(builder.getInt32Ty());
    // 3. Data length (number of elements in source arrays), by value
    wrapper_arg_types.push_back(builder.getInt32Ty());
    // 4. Pointers for each input data array
    for (toolchain::Type* arg_vec_type : arg_vec_types) {
      wrapper_arg_types.push_back(
          arg_vec_type->getScalarType()->getPointerTo());
    }

    toolchain::FunctionType* wrapper_llvm_func_type =
        toolchain::FunctionType::get(builder.getVoidTy(), wrapper_arg_types, false);
    std::string wrapper_function_name = original_function_name + "_wrapper";
    toolchain::Function* wrapper_llvm_func = toolchain::Function::Create(
        wrapper_llvm_func_type, toolchain::Function::ExternalLinkage,
        wrapper_function_name, *wrapper_module);

    toolchain::BasicBlock* entry_block =
        toolchain::BasicBlock::Create(*ctx, "entry", wrapper_llvm_func);
    toolchain::BasicBlock* loop_header =
        toolchain::BasicBlock::Create(*ctx, "loop.header", wrapper_llvm_func);
    toolchain::BasicBlock* loop_body =
        toolchain::BasicBlock::Create(*ctx, "loop.body", wrapper_llvm_func);
    toolchain::BasicBlock* loop_exit =
        toolchain::BasicBlock::Create(*ctx, "loop.exit", wrapper_llvm_func);
    builder.SetInsertPoint(entry_block);
    // Use a pointer here to make the loop difficult to optimize away.
    toolchain::AllocaInst* counter =
        builder.CreateAlloca(builder.getInt32Ty(), nullptr, "counter");
    builder.CreateStore(builder.getInt32(0), counter);
    builder.CreateBr(loop_header);

    builder.SetInsertPoint(loop_header);
    toolchain::Value* loop_iterations = wrapper_llvm_func->getArg(1);
    toolchain::Value* current_count =
        builder.CreateLoad(builder.getInt32Ty(), counter, "current_count");
    toolchain::Value* condition =
        builder.CreateICmpSLT(current_count, loop_iterations, "loop_cond");
    builder.CreateCondBr(condition, loop_body, loop_exit);

    builder.SetInsertPoint(loop_body);
    toolchain::Value* data_length = wrapper_llvm_func->getArg(2);

    // Calculate the number of vectors in the data array
    toolchain::Value* num_vectors = builder.CreateUDiv(
        data_length, builder.getInt32(vector_size), "num_vectors");
    // Calculate the index for this iteration: current_count % num_vectors
    toolchain::Value* index =
        builder.CreateURem(current_count, num_vectors, "index");

    // Load input vectors using the calculated index
    std::vector<toolchain::Value*> arg_vecs;
    for (int i = 0; i < arg_types.size(); ++i) {
      toolchain::Value* base_ptr = wrapper_llvm_func->getArg(i + 3);
      toolchain::Value* vec_ptr =
          builder.CreateGEP(arg_vec_types[i], base_ptr, index, "vec_ptr");
      toolchain::LoadInst* arg_vec =
          builder.CreateLoad(arg_vec_types[i], vec_ptr, "arg_vec");
      arg_vec->setAlignment(toolchain::Align(32));
      arg_vecs.push_back(arg_vec);
    }
    toolchain::CallInst* result_vec = builder.CreateCall(original_func, arg_vecs);
    toolchain::Value* ret_base_ptr = wrapper_llvm_func->getArg(0);
    toolchain::Value* ret_vec_ptr =
        builder.CreateGEP(ret_vec_type, ret_base_ptr, index, "ret_vec_ptr");
    toolchain::StoreInst* store_result =
        builder.CreateStore(result_vec, ret_vec_ptr);
    store_result->setAlignment(toolchain::Align(32));

    toolchain::Value* next_count =
        builder.CreateAdd(current_count, builder.getInt32(1), "next_count");
    builder.CreateStore(next_count, counter);
    builder.CreateBr(loop_header);
    builder.SetInsertPoint(loop_exit);
    builder.CreateRetVoid();

    std::string error_str;
    toolchain::raw_string_ostream os(error_str);
    if (toolchain::verifyFunction(*wrapper_llvm_func, &os)) {
      return toolchain::make_error<toolchain::StringError>(
          toolchain::errc::invalid_argument,
          "Error in wrapper function IR: " + os.str());
    }
    toolchain::ExitOnError exit_on_err;
    exit_on_err(jit_->addIRModule(
        toolchain::orc::ThreadSafeModule(std::move(wrapper_module_owner), *tsc_)));
    auto function_sym = jit_->lookup(wrapper_function_name);
    if (!function_sym) {
      return function_sym.takeError();
    }
    return reinterpret_cast<void*>(function_sym->getValue());
  });
}

std::unique_ptr<toolchain::TargetMachine> CreateHostTargetMachine() {
  initializeNativeTargets();
  const std::string triple = toolchain::sys::getDefaultTargetTriple();
  toolchain::StringRef cpu = toolchain::sys::getHostCPUName();
  toolchain::StringMap<bool> features = toolchain::sys::getHostCPUFeatures();
  std::string errors = "";
  const toolchain::Target* target =
      toolchain::TargetRegistry::lookupTarget(toolchain::StringRef(triple), errors);
  LOG_IF(FATAL, !target) << "Failed to lookup target: " << errors;
  std::string feature_str;
  for (const auto& [feature, value] : features) {
    if (value) {
      feature_str += "+" + feature.str() + ",";
    }
  }
  toolchain::TargetOptions target_options;
  std::unique_ptr<toolchain::TargetMachine> target_machine(
      target->createTargetMachine(toolchain::Triple(triple), cpu, feature_str,
                                  target_options, std::nullopt, std::nullopt));
  LOG_IF(FATAL, !target_machine) << "Failed to create target machine";
  return target_machine;
}
}  // namespace xla::codegen::intrinsic
