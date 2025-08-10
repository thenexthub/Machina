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

#ifndef MACHINA_SERVING_CORE_TEST_UTIL_MOCK_STORAGE_PATH_TARGET_H_
#define MACHINA_SERVING_CORE_TEST_UTIL_MOCK_STORAGE_PATH_TARGET_H_

#include <vector>

#include <gmock/gmock.h>
#include "machina/core/lib/core/status.h"
#include "machina_serving/core/storage_path.h"
#include "machina_serving/core/target.h"

namespace machina {
namespace serving {
namespace test_util {

class MockStoragePathTarget : public TargetBase<StoragePath> {
 public:
  ~MockStoragePathTarget() override { Detach(); }
  MOCK_METHOD(void, SetAspiredVersions,
              (const StringPiece, std::vector<ServableData<StoragePath>>),
              (override));
};

}  // namespace test_util
}  // namespace serving
}  // namespace machina

#endif  // MACHINA_SERVING_CORE_TEST_UTIL_MOCK_STORAGE_PATH_TARGET_H_
