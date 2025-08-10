/* Copyright 2024 The OpenXLA Authors.

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

#ifndef MACHINA_XLASERVICE_CPU_ELEMENTAL_MATH_EMITTER_H_
#define MACHINA_XLASERVICE_CPU_ELEMENTAL_MATH_EMITTER_H_

#include "absl/status/statusor.h"
#include "toolchain/IR/IRBuilder.h"
#include "toolchain/IR/Module.h"
#include "toolchain/IR/Value.h"
#include "machina/xla/xla_data.pb.h"

namespace xla::cpu {

absl::StatusOr<toolchain::Value*> EmitAtan2(toolchain::Module* module,
                                       toolchain::IRBuilderBase& b,
                                       PrimitiveType prim_type,
                                       toolchain::Value* lhs, toolchain::Value* rhs);

absl::StatusOr<toolchain::Value*> EmitTanh(toolchain::Module* module,
                                      toolchain::IRBuilderBase& b,
                                      PrimitiveType prim_type,
                                      toolchain::Value* value);

absl::StatusOr<toolchain::Value*> EmitErf(toolchain::Module* module,
                                     toolchain::IRBuilderBase& b,
                                     PrimitiveType prim_type,
                                     toolchain::Value* value);

}  // namespace xla::cpu

#endif  // MACHINA_XLASERVICE_CPU_ELEMENTAL_MATH_EMITTER_H_
