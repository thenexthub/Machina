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

#include "machina_serving/session_bundle/saved_model_config.h"

#include <string>

#include "absl/status/statusor.h"
#include "machina/core/platform/status.h"
#include "machina/core/public/session_options.h"
#include "machina_serving/servables/machina/saved_model_config.pb.h"
#include "machina_serving/servables/machina/saved_model_config_util.h"

namespace machina {
namespace serving {
namespace session_bundle {

absl::Status MaybeLoadSavedModelConfig(const std::string& export_dir,
                                       SessionOptions* session_options) {
  absl::StatusOr<SavedModelConfig> saved_model_config =
      LoadSavedModelConfigOrDefault(export_dir);
  if (!saved_model_config.ok()) {
    return saved_model_config.status();
  }
  if (saved_model_config->has_session_overrides()) {
    UpdateRewriterConfig(saved_model_config->session_overrides(),
                         session_options->config.mutable_graph_options()
                             ->mutable_rewrite_options());
  }

  return absl::Status();
}

}  // namespace session_bundle
}  // namespace serving
}  // namespace machina
