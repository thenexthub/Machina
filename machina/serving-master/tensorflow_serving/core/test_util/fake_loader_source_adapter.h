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

#ifndef MACHINA_SERVING_CORE_TEST_UTIL_FAKE_LOADER_SOURCE_ADAPTER_H_
#define MACHINA_SERVING_CORE_TEST_UTIL_FAKE_LOADER_SOURCE_ADAPTER_H_

#include <functional>

#include "machina/core/lib/core/status.h"
#include "machina/core/platform/macros.h"
#include "machina/core/platform/types.h"
#include "machina_serving/core/simple_loader.h"
#include "machina_serving/core/storage_path.h"
#include "machina_serving/core/test_util/fake_loader_source_adapter.pb.h"

namespace machina {
namespace serving {
namespace test_util {

// A fake loader source adapter that creates loaders of servable-type string
// from data of type StoragePath.
//
// If path = "/a/simple/path" and suffix = "foo", the servable string becomes
// "a/simple/path/foo".
//
// To help with verifying the order of destruction of these adapters in tests,
// the adapter may take a callback to be invoked upon destruction. The
// suffix provided to the source-adapter is passed to the string argument of the
// callback when it is invoked.
class FakeLoaderSourceAdapter final
    : public SimpleLoaderSourceAdapter<StoragePath, string> {
 public:
  FakeLoaderSourceAdapter(
      const string& suffix = "",
      std::function<void(const string&)> call_on_destruct = {});

  ~FakeLoaderSourceAdapter() override;

 private:
  const string suffix_;
  std::function<void(const string&)> call_on_destruct_;
  TF_DISALLOW_COPY_AND_ASSIGN(FakeLoaderSourceAdapter);
};

}  // namespace test_util
}  // namespace serving
}  // namespace machina

#endif  // MACHINA_SERVING_CORE_TEST_UTIL_FAKE_LOADER_SOURCE_ADAPTER_H_
