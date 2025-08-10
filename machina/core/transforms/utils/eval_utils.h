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

#ifndef MACHINA_CORE_TRANSFORMS_UTILS_EVAL_UTILS_H_
#define MACHINA_CORE_TRANSFORMS_UTILS_EVAL_UTILS_H_

#include <memory>

#include "absl/status/status.h"
#include "toolchain/ADT/ArrayRef.h"
#include "toolchain/ADT/SmallVector.h"
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "machina/core/framework/device_base.h"
#include "machina/core/framework/resource_mgr.h"
#include "machina/core/framework/tensor.h"
#include "machina/core/framework/types.h"
#include "machina/core/ir/tf_op_wrapper.h"

namespace Eigen {
class ThreadPoolDevice;
}  // namespace Eigen

namespace mlir {
namespace tfg {
namespace util {

// A simple CPU device for operation evaluation.
class SimpleDevice : public machina::DeviceBase {
 public:
  SimpleDevice();
  ~SimpleDevice() override;

  absl::Status MakeTensorFromProto(
      const machina::TensorProto& tensor_proto,
      const machina::AllocatorAttributes alloc_attrs,
      machina::Tensor* tensor) override;

  machina::Allocator* GetAllocator(
      machina::AllocatorAttributes attr) override;

  const std::string& device_type() const override { return device_type_; }

 private:
  std::unique_ptr<machina::thread::ThreadPool> eigen_worker_;
  machina::DeviceBase::CpuWorkerThreads eigen_worker_threads_;
  std::unique_ptr<Eigen::ThreadPoolDevice> eigen_device_;
  const std::string device_type_ = machina::DEVICE_CPU;
};

// Attempts to evaluates an MLIR Operation with the op registered kernel. The op
// is always executed on the local host CPU irrespective of the device attribute
// of the given op. The results will be filled in the results vector.
LogicalResult EvaluateOperation(machina::DeviceBase* cpu_device,
                                machina::ResourceMgr* resource_mgr, TFOp op,
                                ArrayRef<ElementsAttr> operands,
                                SmallVectorImpl<TypedAttr>& results);
}  // namespace util
}  // namespace tfg
}  // namespace mlir

#endif  // MACHINA_CORE_TRANSFORMS_UTILS_EVAL_UTILS_H_
