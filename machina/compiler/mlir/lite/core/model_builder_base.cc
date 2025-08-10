/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 8, 2025.
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
#include "machina/compiler/mlir/lite/core/model_builder_base.h"

#include <stddef.h>

#include <memory>
#include <string>

#include "absl/strings/str_cat.h"
#include "machina/compiler/mlir/lite/allocation.h"
#include "machina/compiler/mlir/lite/core/api/error_reporter.h"

namespace tflite {

#ifndef TFLITE_MCU
// Loads a model from `filename`. If `mmap_file` is true then use mmap,
// otherwise make a copy of the model in a buffer.
std::unique_ptr<Allocation> GetAllocationFromFile(
    const char* filename, ErrorReporter* error_reporter) {
  std::unique_ptr<Allocation> allocation;
  if (MMAPAllocation::IsSupported()) {
    allocation = std::make_unique<MMAPAllocation>(filename, error_reporter);
  } else {
    allocation = std::make_unique<FileCopyAllocation>(filename, error_reporter);
  }
  return allocation;
}

// Loads a model from `fd`. If `mmap_file` is true then use mmap,
// otherwise make a copy of the model in a buffer.
std::unique_ptr<Allocation> GetAllocationFromFile(
    int fd, ErrorReporter* error_reporter) {
  std::unique_ptr<Allocation> allocation;
  if (MMAPAllocation::IsSupported()) {
    allocation = std::make_unique<MMAPAllocation>(fd, error_reporter);
  } else {
    allocation = std::make_unique<FileCopyAllocation>(
        absl::StrCat("/proc/self/fd/", fd).c_str(), error_reporter);
  }
  return allocation;
}

#endif

}  // namespace tflite
