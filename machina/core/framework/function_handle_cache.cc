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
#include "machina/core/framework/function_handle_cache.h"

#include "machina/core/lib/gtl/map_util.h"
#include "machina/core/lib/random/random.h"
#include "machina/core/lib/strings/stringprintf.h"

namespace machina {

FunctionHandleCache::FunctionHandleCache(FunctionLibraryRuntime* lib)
    : lib_(lib),
      state_handle_(
          strings::Printf("%lld", static_cast<long long>(random::New64()))) {}

FunctionHandleCache::~FunctionHandleCache() {
  absl::Status s = Clear();
  if (!s.ok()) {
    LOG(ERROR) << "Failed to clear function handle cache: " << s.ToString();
  }
}

absl::Status FunctionHandleCache::Instantiate(
    const string& function_name, AttrSlice attrs,
    FunctionLibraryRuntime::InstantiateOptions options,
    FunctionLibraryRuntime::Handle* handle) {
  string key = Canonicalize(function_name, attrs, options);
  FunctionLibraryRuntime::Handle h;
  {
    tf_shared_lock l(mu_);
    h = gtl::FindWithDefault(handles_, key, kInvalidHandle);
  }
  if (h == kInvalidHandle) {
    options.state_handle = state_handle_;
    TF_RETURN_IF_ERROR(
        lib_->Instantiate(function_name, attrs, options, handle));
    mutex_lock l(mu_);
    handles_[key] = *handle;
  } else {
    *handle = h;
  }
  return absl::OkStatus();
}

absl::Status FunctionHandleCache::Clear() {
  mutex_lock l(mu_);
  for (const auto& entry : handles_) {
    TF_RETURN_IF_ERROR(lib_->ReleaseHandle(entry.second));
  }
  handles_.clear();
  return absl::OkStatus();
}

}  // namespace machina
