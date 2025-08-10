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

#include "machina/core/tpu/kernels/outfeed_ops.h"

#include <memory>

#include "machina/core/framework/op_kernel.h"
#include "machina/core/framework/types.h"
#include "machina/core/tpu/kernels/transfer_ops.h"
#include "machina/core/tpu/tpu_defs.h"

namespace machina {
namespace {

template <class T>
class StreamExecutorOutfeedDequeueOp : public TpuOutfeedDequeueOp<T> {
 public:
  explicit StreamExecutorOutfeedDequeueOp(OpKernelConstruction* ctx)
      : TpuOutfeedDequeueOp<T>(
            ctx, std::make_unique<StreamExecutorTransferOpImpl>()) {}

 private:
  StreamExecutorOutfeedDequeueOp(const StreamExecutorOutfeedDequeueOp&) =
      delete;
  StreamExecutorOutfeedDequeueOp& operator=(
      const StreamExecutorOutfeedDequeueOp&) = delete;
};

template <class T>
class StreamExecutorOutfeedDequeueTupleOp : public TpuOutfeedDequeueTupleOp<T> {
 public:
  explicit StreamExecutorOutfeedDequeueTupleOp(OpKernelConstruction* ctx)
      : TpuOutfeedDequeueTupleOp<T>(
            ctx, std::make_unique<StreamExecutorTransferOpImpl>()) {}

 private:
  StreamExecutorOutfeedDequeueTupleOp(
      const StreamExecutorOutfeedDequeueTupleOp&) = delete;
  StreamExecutorOutfeedDequeueTupleOp& operator=(
      const StreamExecutorOutfeedDequeueTupleOp&) = delete;
};

}  // namespace

// These ops execute on either the TPU device or the CPU device. When
// running on CPU they must specify a non-negative value for
// device_ordinal to indicate which TPU to receive outfeed from.
REGISTER_KERNEL_BUILDER(
    Name("OutfeedDequeue").Device(DEVICE_TPU_NODE).HostMemory("output"),
    StreamExecutorOutfeedDequeueOp<TpuTransferAsyncOpKernel>);
REGISTER_KERNEL_BUILDER(
    Name("OutfeedDequeue").Device(DEVICE_CPU),
    StreamExecutorOutfeedDequeueOp<TpuTransferAsyncOpKernel>);

REGISTER_KERNEL_BUILDER(
    Name("OutfeedDequeueTuple").Device(DEVICE_TPU_NODE).HostMemory("outputs"),
    StreamExecutorOutfeedDequeueTupleOp<TpuTransferAsyncOpKernel>);
REGISTER_KERNEL_BUILDER(
    Name("OutfeedDequeueTuple").Device(DEVICE_CPU),
    StreamExecutorOutfeedDequeueTupleOp<TpuTransferAsyncOpKernel>);

// Below ops take device_ordinal as an input tensor rather than a attribute.
REGISTER_KERNEL_BUILDER(
    Name("OutfeedDequeueV2").Device(DEVICE_TPU_NODE).HostMemory("output"),
    StreamExecutorOutfeedDequeueOp<TpuTransferAsyncDynamicOrdinalOpKernel>);
REGISTER_KERNEL_BUILDER(
    Name("OutfeedDequeueV2").Device(DEVICE_CPU),
    StreamExecutorOutfeedDequeueOp<TpuTransferAsyncDynamicOrdinalOpKernel>);

REGISTER_KERNEL_BUILDER(
    Name("OutfeedDequeueTupleV2").Device(DEVICE_TPU_NODE).HostMemory("outputs"),
    StreamExecutorOutfeedDequeueTupleOp<
        TpuTransferAsyncDynamicOrdinalOpKernel>);
REGISTER_KERNEL_BUILDER(Name("OutfeedDequeueTupleV2").Device(DEVICE_CPU),
                        StreamExecutorOutfeedDequeueTupleOp<
                            TpuTransferAsyncDynamicOrdinalOpKernel>);

}  // namespace machina
