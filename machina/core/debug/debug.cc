/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 26, 2025.
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

#include <memory>

#include "machina/core/common_runtime/debugger_state_interface.h"
#include "machina/core/debug/debugger_state_impl.h"

namespace machina {
namespace {

// Registers a concrete implementation of DebuggerState for use by
// DirectSession.
class DebuggerStateRegistration {
 public:
  static std::unique_ptr<DebuggerStateInterface> CreateDebuggerState(
      const DebugOptions& options) {
    return std::unique_ptr<DebuggerStateInterface>(new DebuggerState(options));
  }

  static std::unique_ptr<DebugGraphDecoratorInterface>
  CreateDebugGraphDecorator(const DebugOptions& options) {
    return std::unique_ptr<DebugGraphDecoratorInterface>(
        new DebugGraphDecorator(options));
  }

  DebuggerStateRegistration() {
    DebuggerStateRegistry::RegisterFactory(CreateDebuggerState);
    DebugGraphDecoratorRegistry::RegisterFactory(CreateDebugGraphDecorator);
  }
};

static DebuggerStateRegistration register_debugger_state_implementation;

}  // end namespace
}  // end namespace machina
