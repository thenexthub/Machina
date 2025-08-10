/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, May 15, 2025.
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

#include "machina/python/eager/pywrap_tensor_conversion.h"

#include "absl/container/flat_hash_map.h"
#include "absl/hash/hash.h"
#include "machina/c/eager/tfe_tensorhandle_internal.h"
#include "machina/core/lib/monitoring/counter.h"
#include "machina/core/platform/logging.h"

namespace machina {

auto* scalar_cache_hits = machina::monitoring::Counter<0>::New(
    "/machina/eager/python/scalar_cache_hits",
    "Number of times a scalar TFE_TensorHandle was retrieved from cache");
auto* scalar_cache_misses = machina::monitoring::Counter<0>::New(
    "/machina/eager/python/scalar_cache_misses",
    "Number of times a scalar TFE_TensorHandle was not available in cache");

TFE_TensorHandleCache* TFE_TensorHandleCache::Get() {
  // TODO(slebedev): link with Context (in context.py) instead of having
  // a static global?
  static auto* cache = new TFE_TensorHandleCache();
  return cache;
}

TFE_TensorHandle* TFE_TensorHandleCache::Lookup(
    PyObject* value, machina::DataType dtype, TFE_Context* ctx,
    absl::string_view device_name) const {
  CHECK_NOTNULL(value);
#ifdef Py_GIL_DISABLED
  absl::MutexLock lock(&mu_);
#endif  // Py_GIL_DISABLED
  const auto it = cache.find(Key{PyObjectPtr{value}, dtype, ctx, device_name});
  if (it == cache.end()) {
    scalar_cache_misses->GetCell()->IncrementBy(1);
    return nullptr;
  }

  scalar_cache_hits->GetCell()->IncrementBy(1);
  auto* h = it->second;
  machina::unwrap(h)->Ref();
  return h;
}

void TFE_TensorHandleCache::Insert(PyObject* value, machina::DataType dtype,
                                   TFE_Context* ctx,
                                   absl::string_view device_name,
                                   TFE_TensorHandle* h) {
  Py_INCREF(value);
  machina::unwrap(h)->Ref();
#ifdef Py_GIL_DISABLED
  absl::MutexLock lock(&mu_);
#endif  // Py_GIL_DISABLED
  cache.emplace(Key{PyObjectPtr{value}, dtype, ctx, device_name}, h);
}

void TFE_TensorHandleCache::Clear() {
#ifdef Py_GIL_DISABLED
  absl::MutexLock lock(&mu_);
#endif  // Py_GIL_DISABLED
  DecrefUnrefAll();
  cache.clear();
}

}  // namespace machina
