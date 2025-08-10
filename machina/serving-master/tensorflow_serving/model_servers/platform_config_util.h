/* Copyright 2016 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef MACHINA_SERVING_MODEL_SERVERS_PLATFORM_CONFIG_UTIL_H_
#define MACHINA_SERVING_MODEL_SERVERS_PLATFORM_CONFIG_UTIL_H_

#include "machina_serving/config/platform_config.pb.h"
#include "machina_serving/servables/machina/session_bundle_config.pb.h"

namespace machina {
namespace serving {

// Creates a PlatformConfigMap containing a single entry with the key as
// kTensorFlowModelPlatform and the value as a SourceAdapter config proto
// for SavedModelBundleSourceAdapter.
PlatformConfigMap CreateTensorFlowPlatformConfigMap(
    const SessionBundleConfig& session_bundle_config);

}  // namespace serving
}  // namespace machina

#endif  // MACHINA_SERVING_MODEL_SERVERS_PLATFORM_CONFIG_UTIL_H_
