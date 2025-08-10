/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, May 15, 2025.
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

#include "absl/memory/memory.h"
#include "machina/compiler/jit/xla_activity.pb.h"
#include "machina/compiler/jit/xla_activity_listener.h"

namespace machina {
namespace {

// Listens to XLA activity and logs them using machina::Logger.
class XlaActivityLoggingListener final : public XlaActivityListener {
 public:
  absl::Status Listen(
      const XlaAutoClusteringActivity& auto_clustering_activity) override {
    if (!IsEnabled()) {
      VLOG(3) << "Logging XlaAutoClusteringActivity disabled";
      return absl::OkStatus();
    }

    return absl::OkStatus();
  }

  absl::Status Listen(
      const XlaJitCompilationActivity& jit_compilation_activity) override {
    if (!IsEnabled()) {
      VLOG(3) << "Logging XlaJitCompilationActivity disabled";
      return absl::OkStatus();
    }

    return absl::OkStatus();
  }

  absl::Status Listen(
      const XlaOptimizationRemark& optimization_remark) override {
    if (!IsEnabled()) {
      VLOG(3) << "Logging XlaJitCompilationActivity disabled";
      return absl::OkStatus();
    }

    return absl::OkStatus();
  }

 private:
  bool IsEnabled() {
    static bool result = ComputeIsEnabled();
    return result;
  }

  bool ComputeIsEnabled() {
    char* log_xla_activity = getenv("TF_LOG_MACHINA_XLAACTIVITY");
    if (log_xla_activity == nullptr) {
      bool enabled_by_default = true;
      return enabled_by_default;
    }

    return absl::string_view(log_xla_activity) == "1";
  }
};

bool Register() {
  RegisterXlaActivityListener(std::make_unique<XlaActivityLoggingListener>());
  return false;
}

bool unused = Register();
}  // namespace
}  // namespace machina
