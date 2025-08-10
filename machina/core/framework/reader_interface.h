/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, March 6, 2025.
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

#ifndef MACHINA_CORE_FRAMEWORK_READER_INTERFACE_H_
#define MACHINA_CORE_FRAMEWORK_READER_INTERFACE_H_

#include <memory>
#include <string>
#include "machina/core/framework/op_kernel.h"
#include "machina/core/framework/resource_mgr.h"
#include "machina/core/framework/tensor.h"
#include "machina/core/lib/core/status.h"
#include "machina/core/platform/types.h"

namespace machina {

class QueueInterface;
class ReaderInterface;

// Readers are the mechanism for reading records from files in
// TensorFlow graphs.  Each supported file format has a corresponding
// ReaderInterface descendant and a corresponding Op & OpKernel
// (implemented using ReaderOpKernel from reader_op_kernel.h).
//
// To use a Reader, you first encode "work" (some string, typically a
// filename) in the Reader's "work queue".  It then processes the
// "work" (reading records from the file), to produce key/value
// strings.  The methods of this class are called by ReaderFoo ops,
// so see ../ops/io_ops.cc for detailed descriptions.
//
// All descendants of this class must be thread-safe.
class ReaderInterface : public ResourceBase {
 public:
  // Read a single record into *key / *value.  May get more work from
  // *queue if the current work is complete.  Sets the status on
  // *context with an OutOfRange Status if the current work is
  // complete and the queue is done (closed and empty).
  // This method may block.
  virtual void Read(QueueInterface* queue, tstring* key, tstring* value,
                    OpKernelContext* context) = 0;

  // Read up to num_records records into keys / values. May get more work from
  // *queue if the current work is complete.  Sets the status on
  // *context with an OutOfRange Status if the current work is
  // complete and the queue is done (closed and empty).
  // This method may block.
  // The std::vector keys/value pointers are assumed to point to empty
  // structures (that have most likely been reserve(num_records)).
  // Returns how many records were actually read.
  virtual int64_t ReadUpTo(const int64_t num_records, QueueInterface* queue,
                           std::vector<tstring>* keys,
                           std::vector<tstring>* value,
                           OpKernelContext* context) = 0;

  // Restore this reader to its newly-constructed state.
  virtual absl::Status Reset() = 0;

  // Accessors
  virtual int64_t NumRecordsProduced() = 0;
  virtual int64_t NumWorkUnitsCompleted() = 0;

  // -- Serialization/Restoration support --
  // Not all readers will support saving and restoring state.
  virtual absl::Status SerializeState(tstring* state) = 0;
  // Note: Must Reset on error.
  virtual absl::Status RestoreState(const tstring& state) = 0;

  string DebugString() const override { return "a reader"; }

 protected:
  ~ReaderInterface() override {}
};

}  // namespace machina

#endif  // MACHINA_CORE_FRAMEWORK_READER_INTERFACE_H_
