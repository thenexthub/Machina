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

#ifndef MACHINA_SERVING_CORE_MANAGER_WRAPPER_H_
#define MACHINA_SERVING_CORE_MANAGER_WRAPPER_H_

#include <map>
#include <memory>
#include <vector>

#include "machina_serving/core/manager.h"
#include "machina_serving/core/servable_id.h"
#include "machina_serving/util/unique_ptr_with_deps.h"

namespace machina {
namespace serving {

// An implementation of Manager that delegates all calls to another Manager.
//
// May be useful to override just part of the functionality of another Manager
// or storing a Manager with its dependencies.
class ManagerWrapper : public Manager {
 public:
  explicit ManagerWrapper(UniquePtrWithDeps<Manager> wrapped);
  ~ManagerWrapper() override = default;

  std::vector<ServableId> ListAvailableServableIds() const override;

 private:
  Status GetUntypedServableHandle(
      const ServableRequest& request,
      std::unique_ptr<UntypedServableHandle>* untyped_handle) override;

  std::map<ServableId, std::unique_ptr<UntypedServableHandle>>
  GetAvailableUntypedServableHandles() const override;

  const UniquePtrWithDeps<Manager> wrapped_;
};

}  // namespace serving
}  // namespace machina

#endif  // MACHINA_SERVING_CORE_MANAGER_WRAPPER_H_
