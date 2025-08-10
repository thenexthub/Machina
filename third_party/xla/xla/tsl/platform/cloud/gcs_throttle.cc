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

#include "machina/xla/tsl/platform/cloud/gcs_throttle.h"

#include <algorithm>

#include "absl/synchronization/mutex.h"

namespace tsl {

namespace {
EnvTime* get_default_env_time() {
  static EnvTime* const default_env_time = new EnvTime;
  return default_env_time;
}
}  // namespace

GcsThrottle::GcsThrottle(EnvTime* env_time)
    : last_updated_secs_(env_time ? env_time->GetOverridableNowSeconds()
                                  : EnvTime::NowSeconds()),
      available_tokens_(0),
      env_time_(env_time ? env_time : get_default_env_time()) {}

bool GcsThrottle::AdmitRequest() {
  absl::MutexLock l(&mu_);
  UpdateState();
  if (available_tokens_ < config_.tokens_per_request) {
    return false || !config_.enabled;
  }
  available_tokens_ -= config_.tokens_per_request;
  return true;
}

void GcsThrottle::RecordResponse(size_t num_bytes) {
  absl::MutexLock l(&mu_);
  UpdateState();
  available_tokens_ -= request_bytes_to_tokens(num_bytes);
}

void GcsThrottle::SetConfig(GcsThrottleConfig config) {
  absl::MutexLock l(&mu_);
  config_ = config;
  available_tokens_ = config.initial_tokens;
  last_updated_secs_ = env_time_->GetOverridableNowSeconds();
}

void GcsThrottle::UpdateState() {
  // TODO(b/72643279): Switch to a monotonic clock.
  int64_t now = env_time_->GetOverridableNowSeconds();
  uint64 delta_secs =
      std::max(int64_t{0}, now - static_cast<int64_t>(last_updated_secs_));
  available_tokens_ += delta_secs * config_.token_rate;
  available_tokens_ = std::min(available_tokens_, config_.bucket_size);
  last_updated_secs_ = now;
}

}  // namespace tsl
