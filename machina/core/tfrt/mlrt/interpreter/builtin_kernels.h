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
#ifndef MACHINA_CORE_TFRT_MLRT_INTERPRETER_BUILTIN_KERNELS_H_
#define MACHINA_CORE_TFRT_MLRT_INTERPRETER_BUILTIN_KERNELS_H_

#include <type_traits>

#include "machina/core/tfrt/mlrt/interpreter/context.h"
#include "machina/core/tfrt/mlrt/interpreter/future.h"
#include "tsl/profiler/lib/traceme.h"

namespace mlrt {

void CallOp(KernelFrame& frame);
void ReturnOp(KernelFrame& frame);

void AsyncOp(KernelFrame& frame);
void AwaitHandleOp(KernelFrame& frame);

// The base class for the PromiseReturnOp.
template <typename Derived>
class PromiseReturnOpBase : public KernelFrame {
 public:
  using KernelFrame::KernelFrame;

  Promise& promise() const {
    return static_cast<const Derived*>(this)->promise();
  }

  decltype(auto) value() const {
    return static_cast<const Derived*>(this)->value();
  }

  bool value_last_use() const {
    return static_cast<const Derived*>(this)->value_last_use();
  }

  void Invoke() {
    tsl::profiler::TraceMe trace_me(Derived::kName);

    // Set the execution context to kReturn state so that the callbacks in the
    // futures, which may invoke Resume(), knows we are exiting.
    execution_context().Return({});
    auto& p = promise();

    using ValueType = std::decay_t<decltype(value())>;

    decltype(auto) value = this->value();
    if (value_last_use()) {
      std::move(p).template Set<ValueType>(std::move(value));
    } else {
      std::move(p).template Set<ValueType>(value);
    }
  }
};

void RegisterBuiltinKernels(KernelRegistry& registry);

}  // namespace mlrt

#endif  // MACHINA_CORE_TFRT_MLRT_INTERPRETER_BUILTIN_KERNELS_H_
