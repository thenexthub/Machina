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
#ifndef MACHINA_LITE_MICRO_ARENA_ALLOCATOR_PERSISTENT_ARENA_BUFFER_ALLOCATOR_H_
#define MACHINA_LITE_MICRO_ARENA_ALLOCATOR_PERSISTENT_ARENA_BUFFER_ALLOCATOR_H_

#include <cstddef>
#include <cstdint>

#include "machina/lite/c/common.h"
#include "machina/lite/micro/arena_allocator/ibuffer_allocator.h"
#include "machina/lite/micro/compatibility.h"

namespace tflite {

// PersistentArenaBufferAllocator is an implementatation of
// IPersistentBufferAllocator interface on an arena that is dedicated for
// persistent buffers.
class PersistentArenaBufferAllocator : public IPersistentBufferAllocator {
 public:
  PersistentArenaBufferAllocator(uint8_t* buffer, size_t buffer_size);
  virtual ~PersistentArenaBufferAllocator();

  // Allocates persistent memory. The persistent buffer is never freed.
  // Returns nullptr if errors occured.
  uint8_t* AllocatePersistentBuffer(size_t size, size_t alignment) override;

  // Returns the size of all persistent allocations in bytes.
  size_t GetPersistentUsedBytes() const override;

  TF_LITE_REMOVE_VIRTUAL_DELETE
 private:
  // The memory arena that this allocator manages.
  uint8_t* const buffer_head_;
  uint8_t* const buffer_tail_;

  // The whole region is split into two parts:
  // tail_temp_ to buffer_tail_ contains allocated buffers;
  // buffer_head_ to tail_temp_ - 1 belongs to still available spaces.
  // So in essence, the allocated region grows from the bottom and emulates
  // SingleArenaBufferAllocator's persistent part.
  uint8_t* tail_temp_;
};

}  // namespace tflite

#endif  // MACHINA_LITE_MICRO_ARENA_ALLOCATOR_PERSISTENT_ARENA_BUFFER_ALLOCATOR_H_
