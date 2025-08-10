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
#ifndef MACHINA_CORE_TFRT_COMMON_PJRT_CLIENT_FACTORY_REGISTRY_H_
#define MACHINA_CORE_TFRT_COMMON_PJRT_CLIENT_FACTORY_REGISTRY_H_

#include <functional>
#include <memory>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "machina/xla/pjrt/pjrt_client.h"
#include "machina/xla/tsl/framework/device_type.h"
#include "machina/core/framework/registration/registration.h"
#include "machina/core/platform/mutex.h"
#include "machina/core/tfrt/common/pjrt_client_factory_options.h"
#include "tsl/platform/thread_annotations.h"

namespace xla {

using PjrtClientFactory =
    std::function<absl::StatusOr<std::unique_ptr<PjRtClient>>(
        const PjrtClientFactoryOptions&)>;

// The Pjrt client factory registry holds all the registered client factories.
class PjrtClientFactoryRegistry {
 public:
  explicit PjrtClientFactoryRegistry() = default;

  // Registers PjrtClientFactory with DeviceType as key.
  machina::InitOnStartupMarker RegisterPjrtClientFactory(
      const tsl::DeviceType& device_type,
      const PjrtClientFactory& client_factory);

  // Given the device type, finds related PjrtClientFactory function which takes
  // factory option and returns PjrtClient if succeeds.
  absl::StatusOr<std::unique_ptr<PjRtClient>> GetPjrtClient(
      const tsl::DeviceType& device_type,
      const PjrtClientFactoryOptions& options);

  // Returns singleton instance of PjrtClientFactoryRegistry class.
  static PjrtClientFactoryRegistry& Get();

 private:
  absl::flat_hash_map<std::string, PjrtClientFactory> registry_
      TF_GUARDED_BY(mu_);

  mutable ::machina::mutex mu_;
};

// The `REGISTER_PJRT_CLIENT_FACTORY()` calls RegisterPjrtClientFactory on
// program startup.
#define REGISTER_PJRT_CLIENT_FACTORY(pjrt_client, device_type, client_factory) \
  static ::machina::InitOnStartupMarker const register_##pjrt_client =      \
      ::machina::InitOnStartupMarker{}                                      \
      << ::xla::PjrtClientFactoryRegistry::Get().RegisterPjrtClientFactory(    \
             tsl::DeviceType(device_type), client_factory)

}  // namespace xla

#endif  // MACHINA_CORE_TFRT_COMMON_PJRT_CLIENT_FACTORY_REGISTRY_H_
