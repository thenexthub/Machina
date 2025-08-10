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

#ifndef MACHINA_SERVING_CORE_TEST_UTIL_MANAGER_TEST_UTIL_H_
#define MACHINA_SERVING_CORE_TEST_UTIL_MANAGER_TEST_UTIL_H_

#include "machina_serving/core/aspired_versions_manager.h"
#include "machina_serving/core/caching_manager.h"

namespace machina {
namespace serving {
namespace test_util {

// A test utility that provides access to private AspiredVersionsManager
// members.
class AspiredVersionsManagerTestAccess {
 public:
  explicit AspiredVersionsManagerTestAccess(AspiredVersionsManager* manager);

  // Invokes FlushServables() on the manager.
  void FlushServables();

  // Invokes HandlePendingAspiredVersionsRequests() on the manager.
  void HandlePendingAspiredVersionsRequests();

  // Invokes InvokePolicyAndExecuteAction() on the manager.
  void InvokePolicyAndExecuteAction();

  void SetNumLoadThreads(uint32 num_load_threads);

  uint32 num_load_threads() const;

  void SetCustomSortActions(
      AspiredVersionsManager::CustomSortActionsFn custom_sort_actions);

 private:
  AspiredVersionsManager* const manager_;

  TF_DISALLOW_COPY_AND_ASSIGN(AspiredVersionsManagerTestAccess);
};

// A test utility that provides access to private BasicManager members.
class BasicManagerTestAccess {
 public:
  explicit BasicManagerTestAccess(BasicManager* manager);

  void SetNumLoadThreads(uint32 num_load_threads);

  uint32 num_load_threads() const;

 private:
  BasicManager* const manager_;

  TF_DISALLOW_COPY_AND_ASSIGN(BasicManagerTestAccess);
};

// A test utility that provides access to private CachingManager members.
class CachingManagerTestAccess {
 public:
  explicit CachingManagerTestAccess(CachingManager* manager);

  // Returns the size of the load-mutex map that stores the mutex reference per
  // servable-id requested for load.
  int64_t GetLoadMutexMapSize() const;

 private:
  CachingManager* const manager_;

  TF_DISALLOW_COPY_AND_ASSIGN(CachingManagerTestAccess);
};

}  // namespace test_util
}  // namespace serving
}  // namespace machina

#endif  // MACHINA_SERVING_CORE_TEST_UTIL_MANAGER_TEST_UTIL_H_
