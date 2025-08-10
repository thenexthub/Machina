/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Saturday, June 21, 2025.
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
#ifndef MACHINA_COMPILER_MLIR_LITE_METRICS_ERROR_COLLECTOR_H_
#define MACHINA_COMPILER_MLIR_LITE_METRICS_ERROR_COLLECTOR_H_

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "machina/compiler/mlir/lite/metrics/converter_error_data.pb.h"
#include "machina/compiler/mlir/lite/metrics/types_util.h"

namespace mlir {
namespace TFL {

// A singleton to store errors collected by the instrumentation.
class ErrorCollector {
  using ConverterErrorData = tflite::metrics::ConverterErrorData;
  using ConverterErrorDataSet =
      std::unordered_set<ConverterErrorData, ConverterErrorDataHash,
                         ConverterErrorDataComparison>;

 public:
  const ConverterErrorDataSet &CollectedErrors() { return collected_errors_; }

  void ReportError(const ConverterErrorData &error) {
    collected_errors_.insert(error);
  }

  // Clear the set of collected errors.
  void Clear() { collected_errors_.clear(); }

  // Returns the global instance of ErrorCollector.
  static ErrorCollector* GetErrorCollector();

 private:
  ErrorCollector() {}

  ConverterErrorDataSet collected_errors_;

  static ErrorCollector* error_collector_instance_;
};

}  // namespace TFL
}  // namespace mlir
#endif  // MACHINA_COMPILER_MLIR_LITE_METRICS_ERROR_COLLECTOR_H_
