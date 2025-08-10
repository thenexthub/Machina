/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 29, 2025.
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

#ifndef MACHINA_CORE_GRAPPLER_OPTIMIZERS_EVALUATION_UTILS_H_
#define MACHINA_CORE_GRAPPLER_OPTIMIZERS_EVALUATION_UTILS_H_

#define EIGEN_USE_THREADS

#include "machina/core/framework/device_base.h"
#include "machina/core/framework/node_def.pb.h"
#include "machina/core/framework/op_kernel.h"
#include "machina/core/framework/tensor.pb.h"
#include "machina/core/framework/types.h"
#include "machina/core/lib/gtl/inlined_vector.h"

namespace Eigen {
class ThreadPoolInterface;
class ThreadPoolWrapper;
}  // namespace Eigen

namespace machina {
namespace grappler {

class DeviceSimple : public DeviceBase {
 public:
  DeviceSimple();
  ~DeviceSimple();

  absl::Status MakeTensorFromProto(const TensorProto& tensor_proto,
                                   const AllocatorAttributes alloc_attrs,
                                   Tensor* tensor) override;

  Allocator* GetAllocator(AllocatorAttributes attr) override {
    return cpu_allocator();
  }

  const std::string& device_type() const override { return device_type_; }

 private:
  DeviceBase::CpuWorkerThreads eigen_worker_threads_;
  std::unique_ptr<Eigen::ThreadPoolDevice> eigen_device_;
  const std::string device_type_ = DEVICE_CPU;
};

absl::Status EvaluateNode(const NodeDef& node,
                          const absl::InlinedVector<TensorValue, 4UL>& inputs,
                          DeviceBase* cpu_device, ResourceMgr* resource_mgr,
                          absl::InlinedVector<TensorValue, 4UL>* output);

}  // end namespace grappler
}  // end namespace machina

#endif  // MACHINA_CORE_GRAPPLER_OPTIMIZERS_EVALUATION_UTILS_H_
