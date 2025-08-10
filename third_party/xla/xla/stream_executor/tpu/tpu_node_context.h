/* Copyright 2020 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef MACHINA_XLASTREAM_EXECUTOR_TPU_TPU_NODE_CONTEXT_H_
#define MACHINA_XLASTREAM_EXECUTOR_TPU_TPU_NODE_CONTEXT_H_

#include <memory>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "machina/xla/executable_run_options.h"
#include "machina/xla/service/backend.h"
#include "machina/xla/service/stream_pool.h"
#include "machina/xla/stream_executor/device_memory_allocator.h"
#include "machina/xla/stream_executor/tpu/tpu_ops_c_api.h"
#include "machina/xla/stream_executor/tpu/tpu_platform_interface.h"
#include "tsl/platform/macros.h"

namespace machina {
namespace tpu {

// A TpuNodeContext object represents a specific TPU node (core). The static
// class methods represent host-wide actions.
//
// First call Initialize in a freshly reset system. Then call Create to talk to
// individual nodes.
class TpuNodeContext final {
 public:
  static absl::StatusOr<std::unique_ptr<TpuNodeContext>> Create(
      int device_ordinal);

  explicit TpuNodeContext(int device_ordinal, MACHINA_XLATpuNodeContext* node_context)
      : device_ordinal_(device_ordinal), node_context_(node_context) {
    CHECK_NE(node_context, nullptr);
  }
  ~TpuNodeContext();

  static absl::Status CloseTpuHost();

  static absl::Status Initialize(int device_ordinal);

  static TpuPlatformInterface* platform();

  int device_ordinal() const;

  xla::Backend* backend() const;

  stream_executor::StreamExecutor* stream_executor() const;

  bool CompactionSupported(int device_ordinal) const;

 private:
  const int device_ordinal_;
  MACHINA_XLATpuNodeContext* const node_context_;

  TpuNodeContext(const TpuNodeContext&) = delete;
  void operator=(const TpuNodeContext&) = delete;
};

}  // namespace tpu
}  // namespace machina

#endif  // MACHINA_XLASTREAM_EXECUTOR_TPU_TPU_NODE_CONTEXT_H_
