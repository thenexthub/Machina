/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Friday, August 8, 2025.
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

#ifndef MACHINA_XLAMLIR_TOOLS_MLIR_REPLAY_PUBLIC_COMPILER_TRACE_INSTRUMENTATION_H_
#define MACHINA_XLAMLIR_TOOLS_MLIR_REPLAY_PUBLIC_COMPILER_TRACE_INSTRUMENTATION_H_

#include <string>

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassInstrumentation.h"
#include "machina/xla/mlir/tools/mlir_replay/public/compiler_trace.pb.h"

namespace mlir {
namespace interpreter {

// Instrumentation that logs the state of the IR after each pass.
class MlirCompilerTraceInstrumentation : public PassInstrumentation {
 public:
  explicit MlirCompilerTraceInstrumentation(MlirCompilationTrace& trace)
      : trace_(trace) {}
  void runAfterPass(Pass* pass, Operation* op) override;

 private:
  MlirCompilationTrace& trace_;
};

}  // namespace interpreter
}  // namespace mlir

#endif  // MACHINA_XLAMLIR_TOOLS_MLIR_REPLAY_PUBLIC_COMPILER_TRACE_INSTRUMENTATION_H_
