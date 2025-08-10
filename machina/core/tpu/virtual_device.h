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

#ifndef MACHINA_CORE_TPU_VIRTUAL_DEVICE_H_
#define MACHINA_CORE_TPU_VIRTUAL_DEVICE_H_

#include "absl/status/status.h"
#include "machina/core/common_runtime/device.h"
#include "machina/core/framework/device_attributes.pb.h"

namespace machina {

// A dummy device that exists primarily for operator placement, without
// corresponding directly to a piece of hardware.
class VirtualDevice : public Device {
 public:
  VirtualDevice(Env* env, const DeviceAttributes& device_attributes);

  absl::Status Sync() override;
  Allocator* GetAllocator(AllocatorAttributes attr) override;
  absl::Status MakeTensorFromProto(const TensorProto& tensor_proto,
                                   const AllocatorAttributes alloc_attrs,
                                   Tensor* tensor) override;
  absl::Status TryGetDeviceContext(DeviceContext** out_context) override;
};

}  // namespace machina

#endif  // MACHINA_CORE_TPU_VIRTUAL_DEVICE_H_
