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

#ifndef MACHINA_LITE_MICRO_ARENA_ALLOCATOR_RECORDING_SINGLE_ARENA_BUFFER_ALLOCATOR_H_
#define MACHINA_LITE_MICRO_ARENA_ALLOCATOR_RECORDING_SINGLE_ARENA_BUFFER_ALLOCATOR_H_

#include "machina/lite/micro/arena_allocator/single_arena_buffer_allocator.h"
#include "machina/lite/micro/compatibility.h"

namespace tflite {

// Utility class used to log allocations of a SingleArenaBufferAllocator. Should
// only be used in debug/evaluation settings or unit tests to evaluate
// allocation usage.
class RecordingSingleArenaBufferAllocator : public SingleArenaBufferAllocator {
 public:
  RecordingSingleArenaBufferAllocator(uint8_t* buffer_head, size_t buffer_size);
  // TODO(b/157615197): Cleanup constructors/destructor and use factory
  // functions.
  ~RecordingSingleArenaBufferAllocator() override;

  static RecordingSingleArenaBufferAllocator* Create(uint8_t* buffer_head,
                                                     size_t buffer_size);

  // Returns the number of bytes requested from the head or tail.
  size_t GetRequestedBytes() const;

  // Returns the number of bytes actually allocated from the head or tail. This
  // value will be >= to the number of requested bytes due to padding and
  // alignment.
  size_t GetUsedBytes() const;

  // Returns the number of alloc calls from the head or tail.
  size_t GetAllocatedCount() const;

  TfLiteStatus ResizeBuffer(uint8_t* resizable_buf, size_t size,
                            size_t alignment) override;
  uint8_t* AllocatePersistentBuffer(size_t size, size_t alignment) override;

 private:
  size_t requested_head_bytes_;
  size_t requested_tail_bytes_;
  size_t used_bytes_;
  size_t alloc_count_;

  TF_LITE_REMOVE_VIRTUAL_DELETE
};

}  // namespace tflite

#endif  // MACHINA_LITE_MICRO_ARENA_ALLOCATOR_RECORDING_SINGLE_ARENA_BUFFER_ALLOCATOR_H_
