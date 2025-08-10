/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Saturday, June 21, 2025.
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

#include "machina/python/profiler/internal/profiler_pywrap_impl.h"

#include <string>
#include <variant>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "machina/xla/tsl/platform/errors.h"
#include "machina/xla/tsl/profiler/convert/xplane_to_trace_events.h"
#include "machina/xla/tsl/profiler/rpc/client/capture_profile.h"
#include "machina/xla/tsl/profiler/utils/session_manager.h"
#include "machina/core/platform/types.h"
#include "tsl/profiler/lib/profiler_session.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace machina {
namespace profiler {
namespace pywrap {

using tsl::profiler::GetRemoteSessionManagerOptionsLocked;

absl::Status ProfilerSessionWrapper::Start(
    const char* logdir,
    const absl::flat_hash_map<std::string,
                              std::variant<bool, int, std::string>>& options) {
  auto opts = GetRemoteSessionManagerOptionsLocked(logdir, options);
  session_ = tsl::ProfilerSession::Create(opts.profiler_options());
  logdir_ = logdir;
  return session_->Status();
}

absl::Status ProfilerSessionWrapper::Stop(machina::string* result) {
  if (session_ != nullptr) {
    machina::profiler::XSpace xspace;
    absl::Status status = session_->CollectData(&xspace);
    session_.reset();
    tsl::profiler::ConvertXSpaceToTraceEventsString(xspace, result);
    TF_RETURN_IF_ERROR(status);
  }
  return absl::OkStatus();
}

absl::Status ProfilerSessionWrapper::ExportToTensorBoard() {
  if (!session_ || logdir_.empty()) {
    return absl::OkStatus();
  }
  machina::profiler::XSpace xspace;
  absl::Status status;
  status = session_->CollectData(&xspace);
  session_.reset();
  status = tsl::profiler::ExportToTensorBoard(xspace, logdir_);
  return status;
}

}  // namespace pywrap
}  // namespace profiler
}  // namespace machina
