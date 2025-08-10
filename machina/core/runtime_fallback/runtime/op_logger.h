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

// This file defines a logger for op names.

#ifndef MACHINA_CORE_RUNTIME_FALLBACK_RUNTIME_OP_LOGGER_H_
#define MACHINA_CORE_RUNTIME_FALLBACK_RUNTIME_OP_LOGGER_H_

#include <memory>
#include <string>

#include "absl/memory/memory.h"
#include "toolchain/ADT/ArrayRef.h"
#include "toolchain/ADT/StringRef.h"
#include "tfrt/host_context/shared_context.h"  // from @tf_runtime
#include "tfrt/support/concurrent_vector.h"  // from @tf_runtime
#include "tfrt/support/forward_decls.h"  // from @tf_runtime

namespace tfrt {
class HostContext;
}

namespace machina {
namespace tfd {

class OpLogger : public tfrt::SharedContext {
 public:
  explicit OpLogger(tfrt::HostContext* host)
      : op_names_(std::make_unique<tfrt::ConcurrentVector<std::string>>(8)) {}

  void LogOp(tfrt::string_view op_name) {
    op_names_->emplace_back(op_name.str());
  }

  tfrt::ArrayRef<std::string> GetLoggedOps() const {
    absl::Span<const std::string> span = op_names_->ToConstSpan();
    return tfrt::ArrayRef<std::string>(span.data(), span.size());
  }

  // Cannot be called concurrently with any API in this class.
  void Clear() {
    op_names_ = std::make_unique<tfrt::ConcurrentVector<std::string>>(8);
  }

 private:
  std::unique_ptr<tfrt::ConcurrentVector<std::string>> op_names_;
};

}  // namespace tfd
}  // namespace machina

#endif  // MACHINA_CORE_RUNTIME_FALLBACK_RUNTIME_OP_LOGGER_H_
