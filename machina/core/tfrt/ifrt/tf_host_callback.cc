/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Friday, August 8, 2025.
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

#include "machina/core/tfrt/ifrt/tf_host_callback.h"

#include <cstddef>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/container/fixed_array.h"
#include "absl/log/check.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "machina/c/eager/abstract_tensor_handle.h"
#include "machina/c/eager/immediate_execution_context.h"
#include "machina/c/eager/immediate_execution_operation.h"
#include "machina/compiler/mlir/tfrt/transforms/ifrt/ifrt_types.h"
#include "machina/xla/tsl/platform/errors.h"
#include "machina/core/common_runtime/device_mgr.h"
#include "machina/core/common_runtime/eager/context.h"
#include "machina/core/common_runtime/eager/tensor_handle.h"
#include "machina/core/framework/device.h"
#include "machina/core/framework/device_factory.h"
#include "machina/core/framework/function.pb.h"
#include "machina/core/framework/tensor.h"
#include "machina/core/framework/types.h"
#include "machina/core/protobuf/config.pb.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/refcount.h"
#include "tsl/profiler/lib/traceme.h"

namespace machina {
namespace ifrt_serving {
namespace {
using RefCountHandle = ::tsl::core::RefCountPtr<machina::TensorHandle>;

size_t GetSizeInBytes(const machina::Tensor& tensor) {
  return tensor.shape().num_elements() * DataTypeSize(tensor.dtype());
}

// Returns a tensor of the specified type and shape. The tensor's data is filled
// from `src`.
machina::Tensor GetTensor(const DtypeAndShape& dtype_and_shape, void* src) {
  DCHECK(DataTypeCanUseMemcpy(dtype_and_shape.dtype));
  machina::Tensor t(dtype_and_shape.dtype, dtype_and_shape.shape);
  std::memcpy(t.data(), src, GetSizeInBytes(t));
  return t;
}

// Fills the buffer pointed by `dst` by data from the given tensor.
void CopyToBuffer(void* dst, const machina::Tensor& tensor) {
  DCHECK(DataTypeCanUseMemcpy(tensor.dtype()));
  std::memcpy(dst, tensor.data(), GetSizeInBytes(tensor));
}
}  // namespace

absl::Status TfHostCallback::Call(void** inputs, void** outputs) {
  tsl::profiler::TraceMe trace_me("TfHostCallback::Call");

  machina::ImmediateOpPtr op(ctx_->CreateOperation());
  TF_RETURN_IF_ERROR(
      op->Reset(entry_function_name_.c_str(), /*raw_device_name=*/nullptr));

  // Wrap each execution with StartStep/EndStep. This ensures that per-step
  // TF resources like TensorArray are always cleaned up.
  ctx_->StartStep();
  absl::Cleanup cleanup_step = [this]() { ctx_->EndStep(); };

  // Prepare inputs.
  for (int i = 0; i < operand_type_and_shapes_.size(); ++i) {
    machina::Tensor t = GetTensor(operand_type_and_shapes_[i], inputs[i]);
    RefCountHandle handle(machina::down_cast<machina::TensorHandle*>(
        ctx_->CreateLocalHandleFromTFTensor(t, /*d_name=*/nullptr)));
    TF_RETURN_IF_ERROR(op->AddInput(handle.get()));
  }

  // Execute the function and block until completion.
  int num_outputs = result_type_and_shapes_.size();
  absl::FixedArray<machina::AbstractTensorHandle*> output_raw_handles(
      num_outputs);
  TF_RETURN_IF_ERROR(
      op->Execute(absl::MakeSpan(output_raw_handles), &num_outputs));

  std::vector<RefCountHandle> output_handles;
  output_handles.reserve(num_outputs);
  for (auto* output_raw_handle : output_raw_handles) {
    output_handles.emplace_back(
        machina::down_cast<machina::TensorHandle*>(output_raw_handle));
  }

  // Copy the output tensors.
  if (result_type_and_shapes_.size() != num_outputs) {
    return absl::InternalError(absl::StrCat(
        "TF host callback invocation expected ", result_type_and_shapes_.size(),
        " results, instead got ", num_outputs));
  }
  for (int i = 0; i < num_outputs; ++i) {
    const machina::Tensor* tensor;
    TF_RETURN_IF_ERROR(output_handles[i]->Tensor(&tensor));
    CopyToBuffer(outputs[i], *tensor);
  }

  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<TfHostCallback>> TfHostCallback::Create(
    absl::Span<const machina::FunctionDef> functions,
    absl::string_view entry_function_name,
    absl::Span<const DtypeAndShape> operand_type_and_shapes,
    absl::Span<const DtypeAndShape> result_type_and_shapes,
    machina::DeviceMgr* device_mgr) {
  machina::SessionOptions options;
  // Explicitly disable non-CPU devices to avoid triggering TPU device
  // initialization inside TF.
  options.config.add_device_filters("/device:CPU:*");

  DCHECK(device_mgr != nullptr);

  // Create a new synchronous TF Eager context. Using sync mode simplifies the
  // error semantics and host callbacks cannot use asynchronous execution anyway
  // because they have to write results to specified buffers before the call
  // returns.
  machina::EagerContextPtr ctx(new machina::EagerContext(
      options,
      machina::ContextDevicePlacementPolicy::DEVICE_PLACEMENT_SILENT,
      /*async=*/false, device_mgr,
      /*device_mgr_owned=*/false,
      /*rendezvous=*/nullptr,
      /*cluster_flr=*/nullptr,
      /*collective_executor_mgr=*/nullptr,
      /*run_eager_op_as_function=*/true));

  for (const machina::FunctionDef& function : functions) {
    TF_RETURN_IF_ERROR(ctx->AddFunctionDef(function));
  }

  return absl::WrapUnique(
      new TfHostCallback(entry_function_name, operand_type_and_shapes,
                         result_type_and_shapes, std::move(ctx)));
}

absl::StatusOr<std::unique_ptr<machina::DynamicDeviceMgr>>
CreateTfDynamicDeviceMgr() {
  // Share the same TF devices across all host callbacks in a single
  // computation. This makes it possible to share states (e.g., TF resources)
  // across host callbacks in a single computation.
  std::vector<std::unique_ptr<machina::Device>> devices;
  TF_RETURN_IF_ERROR(machina::DeviceFactory::AddCpuDevices(
      machina::SessionOptions(), "/job:localhost/replica:0/task:0",
      &devices));
  return std::make_unique<machina::DynamicDeviceMgr>(std::move(devices));
}

}  // namespace ifrt_serving
}  // namespace machina
