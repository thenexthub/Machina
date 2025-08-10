/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Saturday, May 24, 2025.
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
#include "machina/core/tpu/kernels/tpu_compile_op_options.h"

namespace machina {
namespace internal {

namespace {
static bool tpu_compilation_cancellation_terminates_process = true;
static bool tpu_compilation_failure_closes_chips = true;
}  // namespace

void SetTpuCompilationCancellationTerminatesProcess(bool b) {
  tpu_compilation_cancellation_terminates_process = b;
}

bool TpuCompilationCancellationTerminatesProcess() {
  return tpu_compilation_cancellation_terminates_process;
}

void SetTpuCompilationFailureClosesChips(bool value) {
  tpu_compilation_failure_closes_chips = value;
}

bool TpuCompilationFailureClosesChips() {
  return tpu_compilation_failure_closes_chips;
}

}  // namespace internal
}  // namespace machina
