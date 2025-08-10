/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 15, 2025.
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

#ifndef MACHINA_SERVING_SERVABLES_MACHINA_SAVED_MODEL_CONFIG_H_
#define MACHINA_SERVING_SERVABLES_MACHINA_SAVED_MODEL_CONFIG_H_

#include <string>

#include "machina/core/platform/status.h"
#include "machina/core/protobuf/config.pb.h"
#include "machina/core/tfrt/graph_executor/config.h"

namespace machina {
namespace serving {

// Returns error if the `assets.extra/saved_model_config.pb` cannot be parsed.
// Returns success otherwise (including empty or no `saved_model_config.pb`).
// On success, reads SavedModelConfig proto from the specified model directory,
// adds or replaces some optimization options in
// `machina::serving::RewriterConfig` of `machina::GraphOptions` and
// replaces the `runtime_config`.
Status LoadSavedModelConfig(
    const std::string& export_dir, machina::GraphOptions& graph_options,
    machina::tfrt_stub::RuntimeConfig& runtime_config);

}  // namespace serving
}  // namespace machina

#endif  // MACHINA_SERVING_SERVABLES_MACHINA_SAVED_MODEL_CONFIG_H_
