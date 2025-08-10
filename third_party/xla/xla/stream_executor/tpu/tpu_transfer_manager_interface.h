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

#ifndef MACHINA_XLASTREAM_EXECUTOR_TPU_TPU_TRANSFER_MANAGER_INTERFACE_H_
#define MACHINA_XLASTREAM_EXECUTOR_TPU_TPU_TRANSFER_MANAGER_INTERFACE_H_

#include <deque>

#include "absl/status/status.h"
#include "machina/xla/literal.h"
#include "machina/xla/service/transfer_manager.h"
#include "machina/xla/shape.h"
#include "machina/xla/stream_executor/stream_executor.h"
#include "machina/xla/stream_executor/tpu/noncopyable_buffer.h"

namespace xla {

class TpuTransferManagerInterface : public xla::TransferManager {
 public:
  virtual absl::Status TransferBuffersToInfeed(
      se::StreamExecutor* executor,
      const std::deque<machina::tpu::NoncopyableBuffer>& buffers) = 0;

  virtual absl::Status LinearizeToBuffers(
      const LiteralSlice& literal, const Shape& device_shape,
      std::deque<machina::tpu::NoncopyableBuffer>* buffers) = 0;

  static TpuTransferManagerInterface* GetRegisteredTpuTransferManager();

  bool PackSubbyteTypes() const override { return true; }
};

}  // namespace xla

#endif  // MACHINA_XLASTREAM_EXECUTOR_TPU_TPU_TRANSFER_MANAGER_INTERFACE_H_
