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
#ifndef MACHINA_CORE_TFRT_FALLBACK_OP_KERNEL_RUNNER_H_
#define MACHINA_CORE_TFRT_FALLBACK_OP_KERNEL_RUNNER_H_

#include <assert.h>
#include <stddef.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "machina/core/common_runtime/device_mgr.h"
#include "machina/core/common_runtime/process_function_library_runtime.h"
#include "machina/core/framework/device.h"
#include "machina/core/framework/node_def.pb.h"
#include "machina/core/framework/op_def.pb.h"
#include "machina/core/framework/op_kernel.h"
#include "machina/core/framework/tensor.h"
#include "machina/core/framework/types.h"
#include "machina/core/platform/errors.h"
#include "machina/core/platform/status.h"

namespace machina {
namespace tfrt_stub {

class OpKernelRunner {
 public:
  static absl::StatusOr<OpKernelRunner> Create(
      absl::string_view op_name, absl::string_view node_name,
      absl::string_view device_name, int num_args,
      const std::function<absl::Status(machina::AttrValueMap*)>&
          attr_builder,
      const machina::DeviceMgr& device_manager,
      const machina::ProcessFunctionLibraryRuntime&
          process_function_library_runtime);

  ABSL_DEPRECATED("Please use the Create() method that takes node_name.")
  static absl::StatusOr<OpKernelRunner> Create(
      absl::string_view op_name, absl::string_view device_name, int num_args,
      const std::function<absl::Status(machina::AttrValueMap*)>&
          attr_builder,
      const machina::DeviceMgr& device_manager,
      const machina::ProcessFunctionLibraryRuntime&
          process_function_library_runtime) {
    return Create(op_name, /*node_name=*/op_name, device_name, num_args,
                  attr_builder, device_manager,
                  process_function_library_runtime);
  }

  static absl::StatusOr<OpKernelRunner> Create(
      absl::string_view op_name, absl::string_view node_name, int num_args,
      const std::function<absl::Status(machina::AttrValueMap*)>&
          attr_builder,
      const machina::ProcessFunctionLibraryRuntime&
          process_function_library_runtime,
      machina::Device* device);

  ABSL_DEPRECATED("Please use the Create() method that takes node_name.")
  static absl::StatusOr<OpKernelRunner> Create(
      absl::string_view op_name, int num_args,
      const std::function<absl::Status(machina::AttrValueMap*)>&
          attr_builder,
      const machina::ProcessFunctionLibraryRuntime&
          process_function_library_runtime,
      machina::Device* device) {
    return Create(op_name, /*node_name=*/op_name, num_args, attr_builder,
                  process_function_library_runtime, device);
  }

  OpKernelRunner() = default;

  explicit operator bool() const { return op_kernel_ != nullptr; }

  void Run(OpKernelContext* context) const {
    DVLOG(1) << "KernelFallbackExecuteCompat Running Op: "
             << op_kernel_->def().DebugString()
             << ", on Device: " << context->device()->name();

    // For TFRT GPU or TPU, we currently only run xla clusters on GPU or TPU,
    // and all other ops are run on CPU.

    op_kernel_->Compute(context);
  }

  void RunAsync(OpKernelContext* context,
                AsyncOpKernel::DoneCallback done_callback) const;

  bool IsAsync() const { return info_->is_async; }

  machina::OpKernel* op_kernel() const { return op_kernel_.get(); }
  machina::Device* device() const { return info_->device; }
  machina::FunctionLibraryRuntime* function_library_runtime() const {
    return info_->function_library_runtime;
  }
  machina::ResourceMgr* resource_manager() const {
    return info_->resource_manager;
  }

  absl::Span<const AllocatorAttributes> input_alloc_attrs() const {
    return input_alloc_attrs_;
  }
  absl::Span<const AllocatorAttributes> output_alloc_attrs() const {
    return output_alloc_attrs_;
  }

 private:
  explicit OpKernelRunner(
      absl::string_view op_name, machina::Device* device,
      machina::FunctionLibraryRuntime* function_library_runtime,
      std::unique_ptr<OpKernel> op_kernel);

  struct Info {
    machina::Device* device = nullptr;
    machina::FunctionLibraryRuntime* function_library_runtime = nullptr;
    machina::ResourceMgr* resource_manager = nullptr;
    bool is_async = false;
    absl::InlinedVector<AllocatorAttributes, 4UL> input_alloc_attrs;
    absl::InlinedVector<AllocatorAttributes, 1UL> output_alloc_attrs;
  };

  std::unique_ptr<OpKernel> op_kernel_;
  absl::Span<const AllocatorAttributes> input_alloc_attrs_;
  std::unique_ptr<Info> info_;
  absl::Span<const AllocatorAttributes> output_alloc_attrs_;

  std::string op_name_;
};

// OpKernelRunState keeps the states needed for per-kernel execution.
struct OpKernelRunState {
  std::vector<const machina::TensorBuffer*> tensor_buffers;
  std::vector<machina::TensorValue> input_tf_tensor_values;
  OpKernelContext::Params params;
  absl::InlinedVector<machina::Tensor, 4UL> input_tf_tensors;

  OpKernelRunState() = default;
  OpKernelRunState(absl::Span<const machina::TensorValue> tensor_values,
                   const OpKernelContext::Params& p,
                   machina::DeviceBase* device = nullptr) {
    // `input_tf_tensor_values` contains the reference to all tensor used,
    // while `input_tf_tensors` only contains those needs ownership so their
    // sizes may not match. For this copy assignment, we conservatively copy all
    // tensors.
    input_tf_tensors.reserve(tensor_values.size());
    for (const auto& tensor_value : tensor_values) {
      input_tf_tensors.push_back(*tensor_value.tensor);
    }
    for (auto& tensor : input_tf_tensors) {
      input_tf_tensor_values.emplace_back(&tensor);
    }

    // Since `input_tf_tensor_values` and `params` contains pointers to
    // `input_tf_tensors`, we need to change those pointers to the correct ones
    // after copying.
    params = p;
    params.inputs = input_tf_tensor_values;
    // Clear eigen_gpu_device to ensure OpKernelContext constructor will make a
    // new eigen GPU device.
    params.eigen_gpu_device = nullptr;
    if (device != nullptr) params.device = device;
  }

  OpKernelRunState(const OpKernelRunState& other) = delete;
  OpKernelRunState& operator=(const OpKernelRunState& other) = delete;

  ~OpKernelRunState() = default;
};

// OpKernelRunnerTable for keeping OpKernelRunner instances to avoid expensive
// reinstantiation of OpKernel and other repeated setup per kernel execution.
// OpKernelRunnerTable is thread-compatible.
class OpKernelRunnerTable {
 public:
  OpKernelRunnerTable() = default;

  // Return true if it successfully inserts `runner`. `index` is supposed to be
  // dense.
  bool Insert(int64_t index, OpKernelRunner runner) {
    if (runners_.size() <= index) runners_.resize(index + 1);
    if (runners_[index]) return false;
    runners_[index] = std::move(runner);
    return true;
  }

  // Return the OpKernelRunner at the corresponding `index` in the table. The
  // result can never be nullptr. It is a fatal error to use an index that is
  // not in the table. Note that the returned pointer will be invalidated if
  // Insert() is called.
  const OpKernelRunner* Get(int64_t index) const {
    // Out of bounds vector access will throw an exception and anyway will crash
    // the binary, prefer a more readable error message.
    CHECK_GT(runners_.size(), index)  // Crash OK
        << "runner index is out of bounds: index=" << index
        << " size=" << runners_.size();
    CHECK(runners_[index])  // Crash OK
        << "runner is not available: index=" << index;
    return GetUnsafe(index);
  }

  const OpKernelRunner* GetUnsafe(int64_t index) const {
    DCHECK_GT(runners_.size(), index);
    auto& result = runners_[index];
    DCHECK(result);
    return &result;
  }

 private:
  std::vector<OpKernelRunner> runners_;
};

}  // namespace tfrt_stub
}  // namespace machina

#endif  // MACHINA_CORE_TFRT_FALLBACK_OP_KERNEL_RUNNER_H_
