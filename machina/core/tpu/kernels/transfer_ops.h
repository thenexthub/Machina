/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Saturday, May 24, 2025.
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

#ifndef MACHINA_CORE_TPU_KERNELS_TRANSFER_OPS_H_
#define MACHINA_CORE_TPU_KERNELS_TRANSFER_OPS_H_

#include <deque>
#include <memory>
#include <string>

#include "machina/xla/literal.h"
#include "machina/xla/stream_executor/stream_executor.h"
#include "machina/xla/stream_executor/tpu/noncopyable_buffer.h"
#include "machina/xla/stream_executor/tpu/tpu_platform_interface.h"
#include "machina/xla/stream_executor/tpu/tpu_transfer_manager_interface.h"
#include "machina/core/framework/op_kernel.h"
#include "machina/core/platform/mutex.h"
#include "machina/core/platform/status.h"
#include "machina/core/platform/statusor.h"
#include "machina/core/platform/threadpool.h"

namespace machina {

class TpuTransferOpInterface {
 public:
  virtual ~TpuTransferOpInterface() = default;
  virtual void Cancel() = 0;
  virtual absl::StatusOr<int> GetDeviceOrdinal(OpKernelContext* ctx) = 0;

  virtual absl::Status TransferBuffersToInfeed(
      int device_ordinal,
      const std::deque<machina::tpu::NoncopyableBuffer>& buffers) = 0;
  virtual absl::Status TransferLiteralToInfeed(
      int device_ordinal, const xla::LiteralSlice& literal) = 0;
  virtual absl::Status TransferLiteralFromOutfeed(
      int device_ordinal, xla::MutableBorrowingLiteral literal) = 0;
};

// Base class providing common functionality for async ops that transfer from
// host to TPU.
class TpuTransferAsyncOpKernelBase : public AsyncOpKernel {
 public:
  explicit TpuTransferAsyncOpKernelBase(
      OpKernelConstruction* ctx, const std::string& transfer_type,
      int number_of_threads,
      std::unique_ptr<TpuTransferOpInterface> transfer_op);

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override;

 protected:
  virtual absl::Status DoWork(OpKernelContext* context, int device_ordinal) = 0;

  absl::Status RunTransferWithOrdinal(OpKernelContext* ctx, int device_ordinal);
  std::string transfer_type_;
  std::unique_ptr<TpuTransferOpInterface> transfer_op_;

 private:
  virtual absl::Status RunTransfer(OpKernelContext* ctx) = 0;

  std::unique_ptr<thread::ThreadPool> thread_pool_;
  mutex mu_;

  // TpuTransferAsyncOpKernelBase is neither copyable nor movable.
  TpuTransferAsyncOpKernelBase(const TpuTransferAsyncOpKernelBase&) = delete;
  TpuTransferAsyncOpKernelBase& operator=(const TpuTransferAsyncOpKernelBase&) =
      delete;
};

class TpuTransferAsyncOpKernel : public TpuTransferAsyncOpKernelBase {
 public:
  explicit TpuTransferAsyncOpKernel(
      OpKernelConstruction* ctx, const std::string& transfer_type,
      int number_of_threads,
      std::unique_ptr<TpuTransferOpInterface> transfer_op);

 private:
  absl::Status RunTransfer(OpKernelContext* ctx) override;
  int device_ordinal_;

  // TpuTransferAsyncOpKernel is neither copyable nor movable.
  TpuTransferAsyncOpKernel(const TpuTransferAsyncOpKernel&) = delete;
  TpuTransferAsyncOpKernel& operator=(const TpuTransferAsyncOpKernel&) = delete;
};

class TpuTransferAsyncDynamicOrdinalOpKernel
    : public TpuTransferAsyncOpKernelBase {
 public:
  explicit TpuTransferAsyncDynamicOrdinalOpKernel(
      OpKernelConstruction* ctx, const std::string& transfer_type,
      int number_of_threads,
      std::unique_ptr<TpuTransferOpInterface> transfer_op);

 private:
  absl::Status RunTransfer(OpKernelContext* ctx) override;

  // TpuTransferAsyncDynamicOpKernel is neither copyable nor movable.
  TpuTransferAsyncDynamicOrdinalOpKernel(
      const TpuTransferAsyncDynamicOrdinalOpKernel&) = delete;
  TpuTransferAsyncDynamicOrdinalOpKernel& operator=(
      const TpuTransferAsyncDynamicOrdinalOpKernel&) = delete;
};

class StreamExecutorTransferOpImpl : public TpuTransferOpInterface {
 public:
  explicit StreamExecutorTransferOpImpl();
  ~StreamExecutorTransferOpImpl() override = default;
  void Cancel() override;
  absl::StatusOr<int> GetDeviceOrdinal(OpKernelContext* ctx) override;

  absl::Status TransferBuffersToInfeed(
      int device_ordinal,
      const std::deque<machina::tpu::NoncopyableBuffer>& buffers) override;
  absl::Status TransferLiteralToInfeed(
      int device_ordinal, const xla::LiteralSlice& literal) override;

  absl::Status TransferLiteralFromOutfeed(
      int device_ordinal, xla::MutableBorrowingLiteral literal) override;

 private:
  absl::StatusOr<stream_executor::StreamExecutor*> GetStreamExecutor(
      int device_ordinal);
  xla::TpuTransferManagerInterface* transfer_manager_;
  tpu::TpuPlatformInterface* tpu_platform_;
};

}  // namespace machina

#endif  // MACHINA_CORE_TPU_KERNELS_TRANSFER_OPS_H_
