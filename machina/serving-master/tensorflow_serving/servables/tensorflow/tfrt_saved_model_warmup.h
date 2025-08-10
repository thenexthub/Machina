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

#ifndef MACHINA_SERVING_SERVABLES_MACHINA_TFRT_SAVED_MODEL_WARMUP_H_
#define MACHINA_SERVING_SERVABLES_MACHINA_TFRT_SAVED_MODEL_WARMUP_H_

#include <string>

#include "machina/cc/saved_model/loader.h"
#include "machina/core/protobuf/saved_model.pb.h"
#include "machina/core/tfrt/saved_model/saved_model.h"
#include "machina_serving/servables/machina/saved_model_warmup_util.h"
#include "machina_serving/servables/machina/session_bundle_config.pb.h"

namespace machina {
namespace serving {

// Run warmup requests to trigger lazy initializations (such as TF
// optimizations, XLA compilations) at load time, and consequently improve first
// request latency.
// Supported request types: Predict.
Status RunSavedModelWarmup(const ModelWarmupOptions& model_warmup_options,
                           const string& export_dir, int lazy_init_threshold,
                           bool skip_warmup_requests_if_initialized,
                           tfrt::SavedModel* saved_model);

}  // namespace serving
}  // namespace machina

#endif  // MACHINA_SERVING_SERVABLES_MACHINA_TFRT_SAVED_MODEL_WARMUP_H_
