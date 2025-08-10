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

#ifndef MACHINA_SERVING_SERVABLES_MACHINA_SAVED_MODEL_CONFIG_UTIL_H_
#define MACHINA_SERVING_SERVABLES_MACHINA_SAVED_MODEL_CONFIG_UTIL_H_

#include <string>

#include "absl/status/statusor.h"
#include "machina/core/protobuf/rewriter_config.pb.h"
#include "machina_serving/servables/machina/saved_model_config.pb.h"

namespace machina {
namespace serving {

// Name of the additional asset file containing a per model configuration proto.
inline constexpr char kSavedModelConfigPath[] = "saved_model_config.pb";

inline constexpr char kRemoteOpConfigRewriter[] = "remote_op_config_rewrite";
inline constexpr char kBatchOpRewriter[] = "batch_op_rewriter";

inline constexpr char kRemoteOpRewriteConfigParamKey[] =
    "remote_op_rewrite_config";
inline constexpr char kBatchOpRewriteConfigParamKey[] =
    "batch_op_rewrite_config";

// Extracts a `SavedModelConfig` proto from the optional asset file in the
// given directory. If the asset file does not exist, it returns an empty
// proto.
absl::StatusOr<SavedModelConfig> LoadSavedModelConfigOrDefault(
    const std::string& export_dir);

// Updates `rewrite_options` based on optimizers options in `session_overrides`.
void UpdateRewriterConfig(
    const machina::serving::SessionOverrides& session_overrides,
    machina::RewriterConfig* rewrite_options);

}  // namespace serving
}  // namespace machina

#endif  //  #define
        //  MACHINA_SERVING_SERVABLES_MACHINA_SAVED_MODEL_CONFIG_UTIL_H_
