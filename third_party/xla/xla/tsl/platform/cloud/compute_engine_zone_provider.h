/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 29, 2025.
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

#ifndef MACHINA_XLATSL_PLATFORM_CLOUD_COMPUTE_ENGINE_ZONE_PROVIDER_H_
#define MACHINA_XLATSL_PLATFORM_CLOUD_COMPUTE_ENGINE_ZONE_PROVIDER_H_

#include "machina/xla/tsl/platform/cloud/compute_engine_metadata_client.h"
#include "machina/xla/tsl/platform/cloud/zone_provider.h"

namespace tsl {

class ComputeEngineZoneProvider : public ZoneProvider {
 public:
  explicit ComputeEngineZoneProvider(
      std::shared_ptr<ComputeEngineMetadataClient> google_metadata_client);
  virtual ~ComputeEngineZoneProvider();

  absl::Status GetZone(string* zone) override;

 private:
  std::shared_ptr<ComputeEngineMetadataClient> google_metadata_client_;
  string cached_zone;
  ComputeEngineZoneProvider(const ComputeEngineZoneProvider&) = delete;
  void operator=(const ComputeEngineZoneProvider&) = delete;
};

}  // namespace tsl

#endif  // MACHINA_XLATSL_PLATFORM_CLOUD_COMPUTE_ENGINE_ZONE_PROVIDER_H_
