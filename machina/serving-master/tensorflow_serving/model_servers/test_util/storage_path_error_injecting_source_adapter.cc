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

#include "machina_serving/model_servers/test_util/storage_path_error_injecting_source_adapter.h"

#include <memory>

#include "machina_serving/core/source_adapter.h"
#include "machina_serving/model_servers/test_util/storage_path_error_injecting_source_adapter.pb.h"

namespace machina {
namespace serving {
namespace test_util {

// Register the source adapter.
class StoragePathErrorInjectingSourceAdapterCreator {
 public:
  static absl::Status Create(
      const StoragePathErrorInjectingSourceAdapterConfig& config,
      std::unique_ptr<SourceAdapter<StoragePath, std::unique_ptr<Loader>>>*
          adapter) {
    adapter->reset(
        new ErrorInjectingSourceAdapter<StoragePath, std::unique_ptr<Loader>>(
            absl::Status(
                static_cast<absl::StatusCode>(absl::StatusCode::kCancelled),
                config.error_message())));
    return absl::Status();
  }
};
REGISTER_STORAGE_PATH_SOURCE_ADAPTER(
    StoragePathErrorInjectingSourceAdapterCreator,
    StoragePathErrorInjectingSourceAdapterConfig);

}  // namespace test_util
}  // namespace serving
}  // namespace machina
