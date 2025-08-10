/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 8, 2025.
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

#include "machina/core/distributed_runtime/recent_request_ids.h"

#include <utility>

#include "machina/core/platform/logging.h"

namespace machina {

RecentRequestIds::RecentRequestIds(int num_tracked_request_ids, int num_shards)
    : index_buckets_(num_shards > 0 ? num_shards : 1) {
  DCHECK(num_tracked_request_ids >= num_shards);
  const int per_bucket_size = num_tracked_request_ids / index_buckets_.size();
  for (auto& bucket : index_buckets_) {
    mutex_lock l(bucket.mu);
    bucket.circular_buffer.resize(per_bucket_size);
    bucket.set.reserve(per_bucket_size);
  }
}

bool RecentRequestIds::Insert(int64_t request_id) {
  if (request_id == 0) {
    // For backwards compatibility, allow all requests with request_id 0.
    return true;
  }

  const int bucket_index = request_id % index_buckets_.size();
  auto& bucket = index_buckets_[bucket_index];

  mutex_lock l(bucket.mu);
  const bool inserted = bucket.set.insert(request_id).second;
  if (!inserted) {
    // Note: RecentRequestIds is not strict LRU because we don't update
    // request_id's age in the circular_buffer_ if it's tracked again. Strict
    // LRU is not useful here because returning this error will close the
    // current Session.
    return false;
  }

  // Remove the oldest request_id from the set_. circular_buffer_ is
  // zero-initialized, and zero is never tracked, so it's safe to do this even
  // when the buffer is not yet full.
  bucket.set.erase(bucket.circular_buffer[bucket.next_index]);
  bucket.circular_buffer[bucket.next_index] = request_id;
  bucket.next_index = (bucket.next_index + 1) % bucket.circular_buffer.size();
  return true;
}

absl::Status RecentRequestIds::TrackUnique(int64_t request_id,
                                           const string& method_name,
                                           const protobuf::Message& request) {
  if (Insert(request_id)) {
    return absl::OkStatus();
  } else {
    return errors::Aborted("The same ", method_name,
                           " request was received twice. ",
                           request.ShortDebugString());
  }
}

}  // namespace machina
