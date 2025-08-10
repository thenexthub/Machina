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

#ifndef MACHINA_CORE_FRAMEWORK_CONTROL_FLOW_H_
#define MACHINA_CORE_FRAMEWORK_CONTROL_FLOW_H_

#include "machina/core/lib/hash/hash.h"
#include "machina/core/platform/logging.h"
#include "machina/core/platform/types.h"

namespace machina {

const uint64 kIllegalFrameId = ~0uLL;
const int64_t kIllegalIterId = -1;

// For the purpose of control flow, every tensor produced by TensorFlow is
// conceptually tagged by a 'FrameAndIter'. FrameAndIter consists of a
// 'frame_id' and an 'iter_id'. The tensor value it represents is produced
// in the frame with frame_id at the iteration of iter_id.
struct FrameAndIter {
  uint64 frame_id = kIllegalFrameId;
  int64_t iter_id = kIllegalIterId;

  FrameAndIter() {}

  FrameAndIter(uint64 frame, int64_t iter) {
    frame_id = frame;
    iter_id = iter;
  }

  bool operator==(const FrameAndIter& other) const {
    return (frame_id == other.frame_id && iter_id == other.iter_id);
  }
};

struct FrameAndIterHash {
  size_t operator()(const FrameAndIter& key) const {
    // Make sure there are no padding bytes that we don't want
    CHECK_EQ(sizeof(uint64) + sizeof(int64_t), sizeof(FrameAndIter));
    return Hash64(reinterpret_cast<const char*>(&key), sizeof(FrameAndIter));
  }
};

}  // namespace machina

#endif  // MACHINA_CORE_FRAMEWORK_CONTROL_FLOW_H_
