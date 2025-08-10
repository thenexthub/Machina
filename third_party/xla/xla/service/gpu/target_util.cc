/* Copyright 2019 The OpenXLA Authors.

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
// Provide helper routine for obtaining  gpu target information useful
// for toolchain IR contruction.

#include "machina/xla/service/gpu/target_util.h"

#include <cstring>
#include <functional>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "toolchain/IR/Attributes.h"
#include "toolchain/IR/CallingConv.h"
#include "toolchain/IR/DerivedTypes.h"
#include "toolchain/IR/FPEnv.h"
#include "toolchain/IR/IRBuilder.h"
#include "toolchain/IR/Instructions.h"
#include "toolchain/IR/Intrinsics.h"
#include "toolchain/IR/IntrinsicsAMDGPU.h"
#include "toolchain/IR/IntrinsicsNVPTX.h"
#include "toolchain/IR/MDBuilder.h"
#include "toolchain/IR/Metadata.h"
#include "toolchain/IR/Module.h"
#include "toolchain/IR/Type.h"
#include "toolchain/IR/Value.h"
#include "toolchain/Support/Casting.h"
#include "toolchain/TargetParser/Triple.h"
#include "machina/xla/hlo/ir/hlo_opcode.h"
#include "machina/xla/primitive_util.h"
#include "machina/xla/service/llvm_ir/llvm_type_conversion_util.h"
#include "machina/xla/service/llvm_ir/llvm_util.h"
#include "machina/xla/util.h"
#include "machina/xla/xla_data.pb.h"
#include "tsl/platform/logging.h"

namespace xla {
namespace gpu {
namespace {
// Utility functions to obtain NVPTX/AMDGPU specific information.
using absl::StrCat;

// Wrapper structure for carrying toolchain intrinsic ids for NVPTX/AMDGPU platforms.
// On AMDGPU, some of these operations are made as device functions instead of
// intrinsics. Therefore a variant type is used to wrap the lambda to call
// those device functions.
struct TargetIntrinsics {
  std::variant<toolchain::Intrinsic::ID,
               std::function<toolchain::CallInst*(toolchain::IRBuilderBase*)>>
      nvptx_intrinsic_or_function;
  std::variant<toolchain::Intrinsic::ID,
               std::function<toolchain::CallInst*(toolchain::IRBuilderBase*)>>
      amdgpu_intrinsic_or_function;
  std::variant<toolchain::Intrinsic::ID,
               std::function<toolchain::CallInst*(toolchain::IRBuilderBase*)>>
      spir_intrinsic_or_function;
};

// Gets the toolchain intrinsic ids on different platforms (NVPTX, AMDGPU)
// corresponding to the give TargetIntrinsicID.
struct TargetIntrinsics GetIntrinsic(TargetIntrinsicID intrin) {
  switch (intrin) {
    case TargetIntrinsicID::kThreadIdx: {
      return {
          toolchain::Intrinsic::nvvm_read_ptx_sreg_tid_x,
          toolchain::Intrinsic::amdgcn_workitem_id_x,
          [](toolchain::IRBuilderBase* b_) -> toolchain::CallInst* {
            return EmitDeviceFunctionCall(
                "_Z32__spirv_BuiltInLocalInvocationIdi", {b_->getInt32(0)},
                {U32}, U64, {b_->getContext()}, b_);
          },
      };
    }
    case TargetIntrinsicID::kThreadIdy: {
      return {
          toolchain::Intrinsic::nvvm_read_ptx_sreg_tid_y,
          toolchain::Intrinsic::amdgcn_workitem_id_y,
          [](toolchain::IRBuilderBase* b_) -> toolchain::CallInst* {
            return EmitDeviceFunctionCall(
                "_Z32__spirv_BuiltInLocalInvocationIdi", {b_->getInt32(1)},
                {U32}, U64, {b_->getContext()}, b_);
          },
      };
    }
    case TargetIntrinsicID::kThreadIdz: {
      return {
          toolchain::Intrinsic::nvvm_read_ptx_sreg_tid_z,
          toolchain::Intrinsic::amdgcn_workitem_id_z,
          [](toolchain::IRBuilderBase* b_) -> toolchain::CallInst* {
            return EmitDeviceFunctionCall(
                "_Z32__spirv_BuiltInLocalInvocationIdi", {b_->getInt32(2)},
                {U32}, U64, {b_->getContext()}, b_);
          },
      };
    }
    case TargetIntrinsicID::kBlockIdx: {
      return {
          toolchain::Intrinsic::nvvm_read_ptx_sreg_ctaid_x,
          toolchain::Intrinsic::amdgcn_workgroup_id_x,
          [](toolchain::IRBuilderBase* b_) -> toolchain::CallInst* {
            return EmitDeviceFunctionCall("_Z26__spirv_BuiltInWorkgroupIdi",
                                          {b_->getInt32(0)}, {U32}, U64,
                                          {b_->getContext()}, b_);
          },
      };
    }
    case TargetIntrinsicID::kBlockIdy: {
      return {
          toolchain::Intrinsic::nvvm_read_ptx_sreg_ctaid_y,
          toolchain::Intrinsic::amdgcn_workgroup_id_y,
          [](toolchain::IRBuilderBase* b_) -> toolchain::CallInst* {
            return EmitDeviceFunctionCall("_Z26__spirv_BuiltInWorkgroupIdi",
                                          {b_->getInt32(1)}, {U32}, U64,
                                          {b_->getContext()}, b_);
          },
      };
    }
    case TargetIntrinsicID::kBlockIdz: {
      return {
          toolchain::Intrinsic::nvvm_read_ptx_sreg_ctaid_z,
          toolchain::Intrinsic::amdgcn_workgroup_id_z,
          [](toolchain::IRBuilderBase* b_) -> toolchain::CallInst* {
            return EmitDeviceFunctionCall("_Z26__spirv_BuiltInWorkgroupIdi",
                                          {b_->getInt32(2)}, {U32}, U64,
                                          {b_->getContext()}, b_);
          },
      };
    }
    case TargetIntrinsicID::kBarrierId: {
      return {[](toolchain::IRBuilderBase* b_) -> toolchain::CallInst* {
                // We need to use the callback mechanism here, because the
                // barrier intrinsics expects a constant 0 as operand, whereas
                // for AMD no operand is expected. We don't want to distinguish
                // at the call site.
                toolchain::Module* module = b_->GetInsertBlock()->getModule();
                toolchain::Function* intrinsic =
                    toolchain::Intrinsic::getOrInsertDeclaration(
                        module,
                        toolchain::Intrinsic::nvvm_barrier_cta_sync_aligned_all, {});
                return b_->CreateCall(intrinsic, {b_->getInt32(0)});
              },
              toolchain::Intrinsic::amdgcn_s_barrier,
              [](toolchain::IRBuilderBase* b_) -> toolchain::CallInst* {
                return EmitDeviceFunctionCall(
                    "_Z22__spirv_ControlBarrierjjj",
                    {b_->getInt32(2), b_->getInt32(2), b_->getInt32(272)},
                    {U32, U32, U32}, U32,
                    toolchain::AttrBuilder(b_->getContext())
                        .addAttribute(toolchain::Attribute::Convergent),
                    b_);
              }};
    }
    case TargetIntrinsicID::kBlockDimx: {
      return {toolchain::Intrinsic::nvvm_read_ptx_sreg_ntid_x,
              [](toolchain::IRBuilderBase* b_) -> toolchain::CallInst* {
                return EmitDeviceFunctionCall("__ockl_get_local_size",
                                              {b_->getInt32(0)}, {U32}, U64,
                                              {b_->getContext()}, b_);
              },
              [](toolchain::IRBuilderBase* b_) -> toolchain::CallInst* {
                return EmitDeviceFunctionCall(
                    "_Z28__spirv_BuiltInWorkgroupSizei", {b_->getInt32(0)},
                    {U32}, U64, {b_->getContext()}, b_);
              }};
    }
    case TargetIntrinsicID::kBlockDimy: {
      return {toolchain::Intrinsic::nvvm_read_ptx_sreg_ntid_y,
              [](toolchain::IRBuilderBase* b_) -> toolchain::CallInst* {
                return EmitDeviceFunctionCall("__ockl_get_local_size",
                                              {b_->getInt32(1)}, {U32}, U64,
                                              {b_->getContext()}, b_);
              },
              [](toolchain::IRBuilderBase* b_) -> toolchain::CallInst* {
                return EmitDeviceFunctionCall(
                    "_Z28__spirv_BuiltInWorkgroupSizei", {b_->getInt32(1)},
                    {U32}, U64, {b_->getContext()}, b_);
              }};
    }
    case TargetIntrinsicID::kBlockDimz: {
      return {toolchain::Intrinsic::nvvm_read_ptx_sreg_ntid_z,
              [](toolchain::IRBuilderBase* b_) -> toolchain::CallInst* {
                return EmitDeviceFunctionCall("__ockl_get_local_size",
                                              {b_->getInt32(2)}, {U32}, U64,
                                              {b_->getContext()}, b_);
              },
              [](toolchain::IRBuilderBase* b_) -> toolchain::CallInst* {
                return EmitDeviceFunctionCall(
                    "_Z28__spirv_BuiltInWorkgroupSizei", {b_->getInt32(2)},
                    {U32}, U64, {b_->getContext()}, b_);
              }};
    }
    case TargetIntrinsicID::kGroupBarrierId: {
      return {toolchain::Intrinsic::nvvm_bar_warp_sync,
              toolchain::Intrinsic::amdgcn_wave_barrier,
              [](toolchain::IRBuilderBase* b_) -> toolchain::CallInst* {
                return EmitDeviceFunctionCall(
                    "_Z22__spirv_ControlBarrierjjj",
                    {b_->getInt32(2), b_->getInt32(2), b_->getInt32(272)},
                    {U32, U32, U32}, U32,
                    toolchain::AttrBuilder(b_->getContext())
                        .addAttribute(toolchain::Attribute::Convergent),
                    b_);
              }};
    }
  }
}

// Wrapper structure for carrying math functions for NVPTX/AMDGPU platforms.
struct TargetDeviceFunction {
  const std::string nvptx_root;
  const std::string amdgpu_root;
  const std::string spir_root;
};

// Gets the device function name on different platforms (NVPTX, AMDGPU)
// corresponding to the given TargetDeviceFunctionID.
struct TargetDeviceFunction GetDeviceFunctionRoot(
    TargetDeviceFunctionID func_id) {
  switch (func_id) {
    case TargetDeviceFunctionID::kAtan2: {
      return {"__nv_atan2", "__ocml_atan2", "_Z17__spirv_ocl_atan2"};
    }
    case TargetDeviceFunctionID::kCos: {
      return {"__nv_cos", "__ocml_cos", "_Z15__spirv_ocl_cos"};
    }
    case TargetDeviceFunctionID::kErf: {
      return {"__nv_erf", "__ocml_erf", "_Z15__spirv_ocl_erf"};
    }
    case TargetDeviceFunctionID::kExp: {
      return {"__nv_exp", "__ocml_exp", "_Z15__spirv_ocl_exp"};
    }
    case TargetDeviceFunctionID::kExpm1: {
      return {"__nv_expm1", "__ocml_expm1", "_Z17__spirv_ocl_expm1"};
    }
    case TargetDeviceFunctionID::kFmod: {
      return {"__nv_fmod", "__ocml_fmod", "_Z16__spirv_ocl_fmod"};
    }
    case TargetDeviceFunctionID::kHypot: {
      return {"__nv_hypot", "__ocml_hypot", "_Z17__spirv_ocl_hypot"};
    }
    case TargetDeviceFunctionID::kLog: {
      return {"__nv_log", "__ocml_log", "_Z15__spirv_ocl_log"};
    }
    case TargetDeviceFunctionID::kLog1p: {
      return {"__nv_log1p", "__ocml_log1p", "_Z17__spirv_ocl_log1p"};
    }
    case TargetDeviceFunctionID::kPow: {
      return {"__nv_pow", "__ocml_pow", "_Z15__spirv_ocl_pow"};
    }
    case TargetDeviceFunctionID::kRsqrt: {
      return {"__nv_rsqrt", "__ocml_rsqrt", "_Z17__spirv_ocl_rsqrt"};
    }
    case TargetDeviceFunctionID::kSin: {
      return {"__nv_sin", "__ocml_sin", "_Z15__spirv_ocl_sin"};
    }
    case TargetDeviceFunctionID::kSqrt: {
      return {"__nv_sqrt", "__ocml_sqrt", "_Z16__spirv_ocl_sqrt"};
    }
    case TargetDeviceFunctionID::kTan: {
      return {"__nv_tan", "__ocml_tan", "_Z15__spirv_ocl_tan"};
    }
    case TargetDeviceFunctionID::kTanh: {
      return {"__nv_tanh", "__ocml_tanh", "_Z16__spirv_ocl_tanh"};
    }
    case TargetDeviceFunctionID::kCbrt: {
      return {"__nv_cbrt", "__ocml_cbrt", "_Z16__spirv_ocl_cbrt"};
    }
  }
}
}  // namespace

std::optional<TargetDeviceFunctionID> GetTargetDeviceFunctionID(HloOpcode op) {
  switch (op) {
    case HloOpcode::kAtan2:
      return TargetDeviceFunctionID::kAtan2;
    case HloOpcode::kCos:
      return TargetDeviceFunctionID::kCos;
    case HloOpcode::kExp:
      return TargetDeviceFunctionID::kExp;
    case HloOpcode::kErf:
      return TargetDeviceFunctionID::kErf;
    case HloOpcode::kExpm1:
      return TargetDeviceFunctionID::kExpm1;
    case HloOpcode::kLog:
      return TargetDeviceFunctionID::kLog;
    case HloOpcode::kLog1p:
      return TargetDeviceFunctionID::kLog1p;
    case HloOpcode::kPower:
      return TargetDeviceFunctionID::kPow;
    case HloOpcode::kRemainder:
      return TargetDeviceFunctionID::kFmod;
    case HloOpcode::kRsqrt:
      return TargetDeviceFunctionID::kRsqrt;
    case HloOpcode::kSin:
      return TargetDeviceFunctionID::kSin;
    case HloOpcode::kSqrt:
      return TargetDeviceFunctionID::kSqrt;
    case HloOpcode::kTan:
      return TargetDeviceFunctionID::kTan;
    case HloOpcode::kTanh:
      return TargetDeviceFunctionID::kTanh;
    case HloOpcode::kCbrt:
      return TargetDeviceFunctionID::kCbrt;
    default:
      break;
  }
  return std::nullopt;
}

namespace {
// TODO(b/370452608): Add more functions that have a fast approximation for f32
// that we can use for f16 types.
bool HasFastF32Approximation(TargetDeviceFunctionID func_id) {
  return func_id == TargetDeviceFunctionID::kExp ||
         func_id == TargetDeviceFunctionID::kLog;
}
}  // namespace

std::string ObtainDeviceFunctionName(TargetDeviceFunctionID func_id,
                                     PrimitiveType output_type,
                                     toolchain::Triple target_triple) {
  // The device math functions differentiate between "double" and "float" by
  // appending a double or float specific suffix to a root name. The suffix and
  // the root name are specific to the target.
  struct TargetDeviceFunction gpu_root_names = GetDeviceFunctionRoot(func_id);
  if (target_triple.isNVPTX()) {
    bool is_supported_output_type =
        output_type == BF16 || output_type == F16 || output_type == F32;
    if (is_supported_output_type) {
      std::string function_name = StrCat(gpu_root_names.nvptx_root, "f");
      if (HasFastF32Approximation(func_id) &&
          (output_type == BF16 || output_type == F16)) {
        // All function names start with "__nv". The approximate version of the
        // function names continues with "_fast".
        return function_name.insert(strlen("__nv"), "_fast");
      }
      return function_name;
    } else if (output_type == F64) {
      return gpu_root_names.nvptx_root;
    } else {
      LOG(FATAL) << "Unexpected type while getting device function name: "
                 << primitive_util::LowercasePrimitiveTypeName(output_type);
    }
  } else if (target_triple.getArch() == toolchain::Triple::amdgcn) {
    // TODO(b/370452608): Are there approximate functions we can use for BF16
    // and F16 types?
    if (output_type == BF16 || output_type == F16 || output_type == F32) {
      return StrCat(gpu_root_names.amdgpu_root, "_f32");
    } else if (output_type == F64) {
      return StrCat(gpu_root_names.amdgpu_root, "_f64");
    } else {
      LOG(FATAL) << "Unexpected type while getting device function name.";
    }
  } else if (target_triple.isSPIR()) {
    // TODO(b/370452608): Are there approximate functions we can use for BF16
    // and F16 types?
    if (output_type == BF16 || output_type == F16 || output_type == F32) {
      if (gpu_root_names.spir_root == "_Z17__spirv_ocl_hypot" ||
          gpu_root_names.spir_root == "_Z15__spirv_ocl_pow" ||
          gpu_root_names.spir_root == "_Z17__spirv_ocl_atan2" ||
          gpu_root_names.spir_root == "_Z16__spirv_ocl_fmod") {
        return StrCat(gpu_root_names.spir_root, "ff");
      } else {
        return StrCat(gpu_root_names.spir_root, "f");
      }
    } else if (output_type == F64) {
      if (gpu_root_names.spir_root == "_Z17__spirv_ocl_hypot" ||
          gpu_root_names.spir_root == "_Z15__spirv_ocl_pow" ||
          gpu_root_names.spir_root == "_Z17__spirv_ocl_atan2" ||
          gpu_root_names.spir_root == "_Z16__spirv_ocl_fmod") {
        return StrCat(gpu_root_names.spir_root, "dd");
      } else {
        return StrCat(gpu_root_names.spir_root, "d");
      }
    } else {
      LOG(FATAL) << "Unexpected type while getting device function name.";
    }
  } else {
    LOG(FATAL) << "Invalid triple " << target_triple.str();
  }
}

toolchain::CallInst* EmitDeviceFunctionCall(
    const std::string& callee_name, absl::Span<toolchain::Value* const> operands,
    absl::Span<const PrimitiveType> input_types, PrimitiveType output_type,
    const toolchain::AttrBuilder& attributes, toolchain::IRBuilderBase* b,
    absl::string_view name) {
  std::vector<toolchain::Type*> ir_input_types;
  toolchain::Module* module = b->GetInsertBlock()->getModule();
  toolchain::Triple target_triple = toolchain::Triple(module->getTargetTriple());
  for (PrimitiveType input_type : input_types) {
    ir_input_types.push_back(
        llvm_ir::PrimitiveTypeToIrType(input_type, b->getContext()));
  }
  toolchain::FunctionType* callee_type = toolchain::FunctionType::get(
      llvm_ir::PrimitiveTypeToIrType(output_type,
                                     b->getContext()),  // Return type.
      ir_input_types,                                   // Parameter types.
      false);  // No variadic arguments.

  // Declares the callee if it is not declared already.
  toolchain::Function* callee = toolchain::dyn_cast<toolchain::Function>(
      b->GetInsertBlock()
          ->getModule()
          ->getOrInsertFunction(callee_name, callee_type)
          .getCallee());

  callee->addFnAttrs(attributes);
  if (target_triple.isSPIR())
    callee->setCallingConv(toolchain::CallingConv::SPIR_FUNC);

  return b->CreateCall(callee, llvm_ir::AsArrayRef(operands), name.data());
}

toolchain::CallInst* EmitCallToTargetIntrinsic(
    TargetIntrinsicID intrinsic_id, absl::Span<toolchain::Value* const> operands,
    absl::Span<toolchain::Type* const> overloaded_types, toolchain::IRBuilderBase* b) {
  toolchain::Module* module = b->GetInsertBlock()->getModule();
  struct TargetIntrinsics gpu_intrinsic_id = GetIntrinsic(intrinsic_id);
  std::variant<toolchain::Intrinsic::ID,
               std::function<toolchain::CallInst*(toolchain::IRBuilderBase*)>>
      llvm_intrinsic_or_function;
  toolchain::Triple target_triple = toolchain::Triple(module->getTargetTriple());
  if (target_triple.isNVPTX()) {
    llvm_intrinsic_or_function = gpu_intrinsic_id.nvptx_intrinsic_or_function;
  } else if (target_triple.getArch() == toolchain::Triple::amdgcn) {
    llvm_intrinsic_or_function = gpu_intrinsic_id.amdgpu_intrinsic_or_function;
  } else if (target_triple.isSPIR()) {
    llvm_intrinsic_or_function = gpu_intrinsic_id.spir_intrinsic_or_function;
  } else {
    LOG(FATAL) << "Invalid triple " << target_triple.str();
  }
  toolchain::Intrinsic::ID* llvm_intrinsic_id_ptr =
      std::get_if<toolchain::Intrinsic::ID>(&llvm_intrinsic_or_function);
  if (llvm_intrinsic_id_ptr) {
    toolchain::Intrinsic::ID llvm_intrinsic_id = *llvm_intrinsic_id_ptr;
    toolchain::Function* intrinsic = toolchain::Intrinsic::getOrInsertDeclaration(
        module, llvm_intrinsic_id, llvm_ir::AsArrayRef(overloaded_types));
    return b->CreateCall(intrinsic, llvm_ir::AsArrayRef(operands));
  }
  std::function<toolchain::CallInst*(toolchain::IRBuilderBase*)>* builder_func =
      std::get_if<std::function<toolchain::CallInst*(toolchain::IRBuilderBase*)>>(
          &llvm_intrinsic_or_function);
  return (*builder_func)(b);
}

void AnnotateFunctionAsGpuKernel(toolchain::Module* module, toolchain::Function* func,
                                 toolchain::IRBuilderBase* b) {
  toolchain::Triple target_triple = toolchain::Triple(module->getTargetTriple());
  if (target_triple.isNVPTX()) {
    // Attach information so NVPTX can recognize function as a CUDA kernel.
    func->setCallingConv(toolchain::CallingConv::PTX_Kernel);

  } else if (target_triple.getArch() == toolchain::Triple::amdgcn) {
    // Attach information so AMDGPU can recognize function as a AMDGPU kernel.
    func->setCallingConv(toolchain::CallingConv::AMDGPU_KERNEL);
    func->addFnAttr("uniform-work-group-size", "true");
  } else if (target_triple.isSPIR()) {
    // Attach information so that it can be recognized as a SPIR kernel.
    func->setCallingConv(toolchain::CallingConv::SPIR_KERNEL);
  } else {
    LOG(FATAL) << "Invalid triple " << target_triple.str();
  }
}

}  // namespace gpu
}  // namespace xla
