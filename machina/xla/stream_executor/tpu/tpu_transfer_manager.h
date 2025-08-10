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

#ifndef MACHINA_MACHINA_XLA_STREAM_EXECUTOR_TPU_TPU_TRANSFER_MANAGER_H_
#define MACHINA_MACHINA_XLA_STREAM_EXECUTOR_TPU_TPU_TRANSFER_MANAGER_H_

#include <cstdint>
#include <deque>
#include <functional>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "machina/xla/literal.h"
#include "machina/xla/service/shaped_buffer.h"
#include "machina/xla/service/transfer_manager.h"
#include "machina/xla/shape.h"
#include "machina/xla/stream_executor/device_memory.h"
#include "machina/xla/stream_executor/platform.h"
#include "machina/xla/stream_executor/stream.h"
#include "machina/xla/stream_executor/stream_executor.h"
#include "machina/xla/stream_executor/tpu/c_api_decl.h"
#include "machina/xla/stream_executor/tpu/noncopyable_buffer.h"
#include "machina/xla/stream_executor/tpu/tpu_executor_c_api.h"
#include "machina/xla/stream_executor/tpu/tpu_transfer_manager_interface.h"

namespace tensorflow {
namespace tpu {

class TpuTransferManager : public xla::TpuTransferManagerInterface {
 public:
  TpuTransferManager();
  ~TpuTransferManager() override;

  stream_executor::Platform::Id PlatformId() const override;

  xla::Shape HostShapeToDeviceShape(
      const xla::Shape& host_shape) const override;

  absl::Status TransferLiteralToDeviceAsync(
      stream_executor::Stream* stream, const xla::LiteralSlice& literal,
      const xla::ShapedBuffer& device_buffer,
      const TransferMetadata* transfer_metadata) override;

  void TransferLiteralFromDevice(
      stream_executor::Stream* stream, const xla::ShapedBuffer& device_buffer,
      xla::MutableBorrowingLiteral literal,
      std::function<void(absl::Status)> done,
      const TransferMetadata* transfer_metadata) override;

  absl::Status TransferLiteralToInfeed(
      stream_executor::StreamExecutor* executor,
      const xla::LiteralSlice& literal) override;

  absl::Status TransferLiteralFromOutfeed(
      stream_executor::StreamExecutor* executor,
      xla::MutableBorrowingLiteral literal) override;

  absl::Status TransferBuffersToInfeed(
      se::StreamExecutor* executor,
      const std::deque<tensorflow::tpu::NoncopyableBuffer>& buffers) override;

  absl::Status ResetDevices(
      absl::Span<stream_executor::StreamExecutor* const> executor) override;

  int64_t GetByteSizeRequirement(const xla::Shape& shape) const override;

  absl::StatusOr<xla::Shape> ChooseCompactLayoutForShape(
      const xla::Shape& host_shape) const override;

  bool CanShapedBufferBeAccessedNow(
      stream_executor::StreamExecutor* executor,
      const xla::ShapedBuffer& device_buffer) const override;

  bool CanBufferBeAccessedNow(
      se::StreamExecutor* executor,
      const se::DeviceMemoryBase& device_buffer) const override;

  absl::Status WriteSingleTupleIndexTable(
      stream_executor::Stream* stream,
      absl::Span<const stream_executor::DeviceMemoryBase> elements,
      const xla::Shape& shape,
      stream_executor::DeviceMemoryBase* region) override;

  absl::Status LinearizeToBuffers(
      const xla::LiteralSlice& literal, const xla::Shape& device_shape,
      std::deque<tensorflow::tpu::NoncopyableBuffer>* buffers) override;

  absl::Status ReadDynamicShapes(se::Stream* stream,
                                 const xla::ShapedBuffer* device_buffer,
                                 xla::Shape* device_shape) override;

 private:
  MACHINA_MACHINA_XLA_TransferManager* manager_;
};

}  // namespace tpu
}  // namespace tensorflow

#endif  // MACHINA_MACHINA_XLA_STREAM_EXECUTOR_TPU_TPU_TRANSFER_MANAGER_H_
