/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, July 12, 2025.
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

#ifndef MACHINA_COMPILER_MLIR_TFRT_TRANSFORMS_IFRT_IFRT_BACKEND_COMPILER_H_
#define MACHINA_COMPILER_MLIR_TFRT_TRANSFORMS_IFRT_IFRT_BACKEND_COMPILER_H_

#include "absl/status/status.h"
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/tfrt/backend_compiler.h"
#include "machina/compiler/mlir/tfrt/transforms/tpu_passes.h"
#include "machina/core/tfrt/runtime/runtime.h"

namespace machina {
namespace ifrt_serving {

// Implements the custom backend compiler for IFRT based serving in TFRT.
class IfrtBackendCompiler : public machina::BackendCompiler {
 public:
  struct Options {
    // If true, disable running TFRTSetTPUDeviceAttrPass which set the default
    // `tf.device` and `device_assignment` attributes.
    // This is a server-level option for now. We can consider to make it a
    // per-model option in the future.
    bool disable_set_default_tpu_device_and_device_assignment_attributes = true;
  };

  explicit IfrtBackendCompiler(TpuCompiler* tpu_compiler = nullptr)
      : tpu_compiler_(tpu_compiler) {}

  explicit IfrtBackendCompiler(const Options& ifrt_backend_compile_options,
                               TpuCompiler* tpu_compiler = nullptr)
      : tpu_compiler_(tpu_compiler),
        compile_options_(ifrt_backend_compile_options) {}

  void GetDependentDialects(mlir::DialectRegistry& registry) const override {
    if (tpu_compiler_) {
      tpu_compiler_->RegisterTPUDialects(&registry);
    }
  }

  // Rewrites the machina graph in MLIR for IFRT serving. The methods
  // extracts regions for IFRT execution on accelerator (e.g. TPU).
  absl::Status CompileTensorflow(
      machina::tfrt_stub::ModelRuntimeContext& model_context,
      mlir::ModuleOp module) const override;

 private:
  TpuCompiler* tpu_compiler_;  // Not owned.
  Options compile_options_;
};

}  // namespace ifrt_serving
}  // namespace machina

#endif  // MACHINA_COMPILER_MLIR_TFRT_TRANSFORMS_IFRT_IFRT_BACKEND_COMPILER_H_
