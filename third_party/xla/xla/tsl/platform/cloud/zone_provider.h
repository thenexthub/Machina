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

#ifndef MACHINA_XLATSL_PLATFORM_CLOUD_ZONE_PROVIDER_H_
#define MACHINA_XLATSL_PLATFORM_CLOUD_ZONE_PROVIDER_H_

#include <string>

#include "machina/xla/tsl/platform/errors.h"
#include "machina/xla/tsl/platform/status.h"

namespace tsl {

/// Interface for a provider of cloud instance zone
class ZoneProvider {
 public:
  virtual ~ZoneProvider() {}

  /// \brief  Gets the zone of the Cloud instance and set the result in `zone`.
  /// Returns OK if success.
  ///
  /// Returns an empty string in the case where the zone does not match the
  /// expected format
  /// Safe for concurrent use by multiple threads.
  virtual absl::Status GetZone(string* zone) = 0;

  static absl::Status GetZone(ZoneProvider* provider, string* zone) {
    if (!provider) {
      return errors::Internal("Zone provider is required.");
    }
    return provider->GetZone(zone);
  }
};

}  // namespace tsl

#endif  // MACHINA_XLATSL_PLATFORM_CLOUD_ZONE_PROVIDER_H_
