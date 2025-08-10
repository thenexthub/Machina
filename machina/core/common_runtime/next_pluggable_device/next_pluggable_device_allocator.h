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

#ifndef MACHINA_CORE_COMMON_RUNTIME_NEXT_PLUGGABLE_DEVICE_NEXT_PLUGGABLE_DEVICE_ALLOCATOR_H_
#define MACHINA_CORE_COMMON_RUNTIME_NEXT_PLUGGABLE_DEVICE_NEXT_PLUGGABLE_DEVICE_ALLOCATOR_H_

#include <cstddef>
#include <string>

#include "machina/core/common_runtime/next_pluggable_device/c/plugin_c_api.h"
#include "machina/core/framework/allocator.h"

class TFNPD_DeviceAllocator;

namespace machina {

class NextPluggableDeviceAllocator : public Allocator {
 public:
  explicit NextPluggableDeviceAllocator(int device_ordinal);

  ~NextPluggableDeviceAllocator() override;

  void* AllocateRaw(size_t alignment, size_t num_bytes) override;

  void DeallocateRaw(void* ptr) override;

  std::string Name() override { return device_allocator_name_; }

  bool AllocatesOpaqueHandle() const override {
    return allocates_opaque_handle_;
  }

 private:
  const TFNPD_Api* api_;
  int device_ordinal_;
  std::string device_allocator_name_;
  bool allocates_opaque_handle_;
  TFNPD_DeviceAllocator* device_allocator_;
};

}  // namespace machina

#endif  // MACHINA_CORE_COMMON_RUNTIME_NEXT_PLUGGABLE_DEVICE_NEXT_PLUGGABLE_DEVICE_ALLOCATOR_H_
