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

#ifndef MACHINA_SERVING_CORE_STATIC_MANAGER_H_
#define MACHINA_SERVING_CORE_STATIC_MANAGER_H_

#include <memory>

#include "machina/core/lib/core/errors.h"
#include "machina_serving/core/basic_manager.h"
#include "machina_serving/core/manager.h"
#include "machina_serving/core/servable_handle.h"
#include "machina_serving/core/servable_id.h"
#include "machina_serving/core/simple_loader.h"

namespace machina {
namespace serving {

// Builds a manager that holds a static set of servables. The result is
// immutable, and cannot be modified after construction.
//
// Typical callers will call AddServable() for each desired servable, and then
// call Build() to produce a Manager.
class StaticManagerBuilder {
 public:
  StaticManagerBuilder();

  // Adds a servable to the manager. Duplicate IDs and null servables will fail
  // to be added and return a failure status.
  template <typename T>
  Status AddServable(const ServableId& id, std::unique_ptr<T> servable);

  // Builds the manager. This builder should not be reused after this.
  std::unique_ptr<Manager> Build();

 private:
  // The manager we are building.
  std::unique_ptr<BasicManager> basic_manager_;

  // The health of the builder.
  Status health_;
};

//
// Implementation details follow. Clients can stop reading.
//

template <typename T>
Status StaticManagerBuilder::AddServable(const ServableId& id,
                                         std::unique_ptr<T> servable) {
  if (servable == nullptr) {
    return errors::InvalidArgument("Servable cannot be null.");
  }
  TF_RETURN_IF_ERROR(health_);
  DCHECK(basic_manager_ != nullptr);

  TF_RETURN_IF_ERROR(basic_manager_->ManageServable(CreateServableData(
      id, std::unique_ptr<Loader>(new SimpleLoader<T>(
              [&servable](std::unique_ptr<T>* const returned_servable) {
                *returned_servable = std::move(servable);
                return Status();
              },
              SimpleLoader<T>::EstimateNoResources())))));
  Status load_status;
  Notification load_done;
  basic_manager_->LoadServable(id, [&](const Status& status) {
    load_status = status;
    load_done.Notify();
  });
  load_done.WaitForNotification();
  return load_status;
}

}  // namespace serving
}  // namespace machina

#endif  // MACHINA_SERVING_CORE_STATIC_MANAGER_H_
