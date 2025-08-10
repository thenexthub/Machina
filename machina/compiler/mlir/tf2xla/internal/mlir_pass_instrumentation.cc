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

#include "machina/compiler/mlir/tf2xla/internal/mlir_pass_instrumentation.h"

#include <algorithm>
#include <functional>
#include <iterator>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/log/log.h"
#include "machina/core/platform/logging.h"

namespace mlir {

class MlirPassInstrumentationRegistry {
 public:
  static MlirPassInstrumentationRegistry& Instance() {
    static MlirPassInstrumentationRegistry* r =
        new MlirPassInstrumentationRegistry;
    return *r;
  }
  std::unordered_map<std::string,
                     std::function<std::unique_ptr<PassInstrumentation>()>>
      instrumentors_;
};

void RegisterPassInstrumentor(
    const std::string& name,
    std::function<std::unique_ptr<PassInstrumentation>()> creator) {
  MlirPassInstrumentationRegistry& r =
      MlirPassInstrumentationRegistry::Instance();
  auto result = r.instrumentors_.emplace(name, creator);
  if (!result.second) {
    VLOG(1) << "Duplicate MLIR pass instrumentor registration";
  }
}

std::vector<std::function<std::unique_ptr<PassInstrumentation>()>>
GetPassInstrumentors() {
  MlirPassInstrumentationRegistry& r =
      MlirPassInstrumentationRegistry::Instance();
  std::vector<std::function<std::unique_ptr<PassInstrumentation>()>> result;
  result.reserve(r.instrumentors_.size());

  std::transform(r.instrumentors_.begin(), r.instrumentors_.end(),
                 std::back_inserter(result), [](auto v) { return v.second; });

  return result;
}

}  // namespace mlir
