/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Friday, April 18, 2025.
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

#ifndef MACHINA_SERVING_CORE_TEST_UTIL_FAKE_STORAGE_PATH_SOURCE_ADAPTER_H_
#define MACHINA_SERVING_CORE_TEST_UTIL_FAKE_STORAGE_PATH_SOURCE_ADAPTER_H_

#include "machina/core/lib/core/status.h"
#include "machina_serving/core/source_adapter.h"
#include "machina_serving/core/storage_path.h"

namespace machina {
namespace serving {
namespace test_util {

// A fake storage-path source-adapter that generates simple string servables
// from paths. If the supplied path is set to "invalid", an invalid-argument
// error is returned upon conversion.
//
// If path = "/a/simple/path" and suffix = "foo", the servable string becomes
// "/a/simple/path/foo".
//
// To help with verifying the order of destruction of these adapters in tests,
// the adapter may take a callback to be invoked upon destruction. The
// suffix provided to the source-adapter is passed to the string argument of the
// callback when it is invoked.
class FakeStoragePathSourceAdapter final
    : public UnarySourceAdapter<StoragePath, StoragePath> {
 public:
  FakeStoragePathSourceAdapter(
      const string& suffix = "",
      std::function<void(const string&)> call_on_destruct = {});

  ~FakeStoragePathSourceAdapter();

 private:
  Status Convert(const StoragePath& data,
                 StoragePath* const converted_data) override;

  const string suffix_;
  std::function<void(const string&)> call_on_destruct_;
};

}  // namespace test_util
}  // namespace serving
}  // namespace machina

#endif  // MACHINA_SERVING_CORE_TEST_UTIL_FAKE_STORAGE_PATH_SOURCE_ADAPTER_H_
