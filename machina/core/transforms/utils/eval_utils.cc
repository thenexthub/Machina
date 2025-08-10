/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, June 15, 2025.
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

#include <memory>

#include "absl/container/inlined_vector.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "machina/core/framework/node_def.pb.h"
#include "machina/core/framework/tensor.pb.h"
#define EIGEN_USE_THREADS

#include "machina/core/transforms/utils/eval_utils.h"

#include <cassert>
#include <utility>

#include "toolchain/ADT/STLExtras.h"
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "machina/core/framework/allocator.h"
#include "machina/core/framework/control_flow.h"
#include "machina/core/framework/op_kernel.h"
#include "machina/core/framework/types.h"
#include "machina/core/ir/importexport/convert_tensor.h"
#include "machina/core/ir/importexport/graphdef_export.h"
#include "machina/core/platform/logging.h"
#include "machina/core/platform/threadpool.h"
#include "machina/core/public/version.h"

namespace mlir {
namespace tfg {
namespace util {

// The SimpleDevice is supposed to be used for evaluating single operation. To
// avoid the overhead of thread creation. Set a small and conservative number as
// the default.
static constexpr int kThreads = 2;

SimpleDevice::SimpleDevice() : DeviceBase(machina::Env::Default()) {
  eigen_worker_ = std::make_unique<machina::thread::ThreadPool>(
      machina::Env::Default(), "eval_utils", kThreads);

  eigen_worker_threads_.num_threads = kThreads;
  eigen_worker_threads_.workers = eigen_worker_.get();

  eigen_device_ = std::make_unique<Eigen::ThreadPoolDevice>(
      eigen_worker_threads_.workers->AsEigenThreadPool(),
      eigen_worker_threads_.num_threads);
  set_machina_cpu_worker_threads(&eigen_worker_threads_);
  set_eigen_cpu_device(eigen_device_.get());
}

SimpleDevice::~SimpleDevice() {}

machina::Allocator *SimpleDevice::GetAllocator(
    machina::AllocatorAttributes attr) {
  return machina::cpu_allocator();
}

absl::Status SimpleDevice::MakeTensorFromProto(
    const machina::TensorProto &tensor_proto,
    const machina::AllocatorAttributes alloc_attrs,
    machina::Tensor *tensor) {
  machina::Tensor parsed(tensor_proto.dtype());
  if (!parsed.FromProto(machina::cpu_allocator(), tensor_proto)) {
    return machina::errors::InvalidArgument(
        "Cannot parse tensor from tensor_proto.");
  }
  *tensor = std::move(parsed);
  return absl::OkStatus();
}

LogicalResult EvaluateOperation(machina::DeviceBase *cpu_device,
                                machina::ResourceMgr *resource_mgr, TFOp op,
                                ArrayRef<ElementsAttr> operands,
                                SmallVectorImpl<TypedAttr> &results) {
  assert(cpu_device && "cpu device can't be null");
  assert(resource_mgr && "ResourceMgr can't be null");

  if (toolchain::any_of(operands, [](Attribute operand) { return !operand; })) {
    VLOG(3) << "cannot be evaluated with null operands";
    return failure();
  }

  machina::NodeDef node_def;
  if (!ConvertToNodeDef(&*op, &node_def, op.getDialect(), [&](Value value) {
         return GetValueName(value, op.getDialect());
       }).ok()) {
    VLOG(3) << "failed to convert operation to NodeDef";
    return failure();
  }

  absl::InlinedVector<machina::Tensor, 4> input_tensors(operands.size());
  absl::InlinedVector<machina::TensorValue, 4> input_tensor_values(
      operands.size());
  // For each operand, convert its ElementsAttr to a Tensor and the Tensor will
  // be referenced by a TensorValue. To ensure Tensor/TensorValue have their
  // lifecycle across the later evaluation. They are stored in
  // `input_tensors`\`input_tensor_values` respectively. The following loop zips
  // them together so that the bundled values are related. Note that the
  // accessor index associates with the order of arguments in toolchain::zip.
  for (auto it : toolchain::zip(operands, input_tensors, input_tensor_values)) {
    auto &[operand, input_tensor, input_tensor_value] = it;
    if (!ConvertToTensor(operand, &input_tensor).ok()) return failure();
    input_tensor_value.tensor = &input_tensor;
  }

  absl::Status status;
  std::unique_ptr<machina::OpKernel> op_kernel = machina::CreateOpKernel(
      machina::DEVICE_CPU, cpu_device, cpu_device->GetAllocator({}),
      node_def, TF_GRAPH_DEF_VERSION, &status);
  if (!status.ok()) {
    VLOG(3) << status.message();
    return failure();
  }

  machina::OpKernelContext::Params params;
  params.device = cpu_device;
  params.frame_iter = machina::FrameAndIter(0, 0);
  params.inputs = input_tensor_values;
  params.op_kernel = op_kernel.get();
  params.resource_manager = resource_mgr;

  absl::InlinedVector<machina::AllocatorAttributes, 4> output_attrs(
      op_kernel->num_outputs());
  for (auto &attr : output_attrs) attr.set_on_host(true);
  params.output_attr_array = output_attrs.data();

  // Evaluate the operation.
  machina::OpKernelContext op_context(&params);
  op_kernel->Compute(&op_context);
  if (!op_context.status().ok()) {
    VLOG(3) << op_context.status().message();
    return failure();
  }

  // Converts the outputs to MLIR attributes.
  Builder builder(op->getContext());
  for (int i = 0; i < op_kernel->num_outputs(); ++i) {
    // The output is invalidated, returns a `dead` value here.
    if (op_context.mutable_output(i) == nullptr) {
      results.push_back(nullptr);
      continue;
    }

    absl::StatusOr<ElementsAttr> attr_or =
        ConvertTensor(*(op_context.mutable_output(i)), builder);
    if (!attr_or.status().ok()) {
      VLOG(3) << attr_or.status().message();
      return failure();
    }
    results.push_back(attr_or.value());
  }

  return success();
}

}  // namespace util
}  // namespace tfg
}  // namespace mlir
