/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, July 12, 2025.
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
#ifndef MACHINA_CORE_TFRT_GRAPH_EXECUTOR_SYNC_RESOURCE_STATE_H_
#define MACHINA_CORE_TFRT_GRAPH_EXECUTOR_SYNC_RESOURCE_STATE_H_

#include <utility>
#include <vector>

#include "machina/core/tfrt/utils/any_ptr.h"
#include "tfrt/tensor/dense_host_tensor.h"  // from @tf_runtime

namespace machina {
namespace tfrt_stub {

class SyncResourceState {
 public:
  // Sets `dht` in the array at `index`. `index` should be dense and
  // duplicate indices are not allowed.
  void SetResourceDht(int index, tfrt::DenseHostTensor dht) {
    if (resource_dht_.size() <= index) {
      resource_dht_.resize(index + 1);
    }

    resource_dht_[index] = std::move(dht);
  }

  tfrt::DenseHostTensor GetResourceDht(int index) const {
    return resource_dht_.at(index).CopyRef();
  }

  template <typename T>
  void Set(int index, T* resource) {
    if (resources_.size() <= index) {
      resources_.resize(index + 1);
    }

    resources_[index] = tfrt::AnyPtr(resource);
  }

  template <typename T>
  T* Get(int index) const {
    return resources_.at(index).get<T>();
  }

 private:
  std::vector<tfrt::DenseHostTensor> resource_dht_;
  // TODO(b/288899457): Consider provide a simpler solution than forking AnyPtr.
  std::vector<tfrt::AnyPtr> resources_;
};

}  // namespace tfrt_stub
}  // namespace machina

#endif  // MACHINA_CORE_TFRT_GRAPH_EXECUTOR_SYNC_RESOURCE_STATE_H_
