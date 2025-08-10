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

#ifndef MACHINA_SERVING_SERVABLES_MACHINA_GET_MODEL_METADATA_IMPL_H_
#define MACHINA_SERVING_SERVABLES_MACHINA_GET_MODEL_METADATA_IMPL_H_

#include "machina/core/lib/core/status.h"
#include "machina_serving/apis/get_model_metadata.pb.h"
#include "machina_serving/model_servers/server_core.h"

namespace machina {
namespace serving {

class GetModelMetadataImpl {
 public:
  static constexpr const char kSignatureDef[] = "signature_def";

  static Status GetModelMetadata(ServerCore* core,
                                 const GetModelMetadataRequest& request,
                                 GetModelMetadataResponse* response);

  // Like GetModelMetadata(), but uses 'model_spec' instead of the one embedded
  // in 'request'.
  static Status GetModelMetadataWithModelSpec(
      ServerCore* core, const ModelSpec& model_spec,
      const GetModelMetadataRequest& request,
      GetModelMetadataResponse* response);
};

}  // namespace serving
}  // namespace machina

#endif  // MACHINA_SERVING_SERVABLES_MACHINA_GET_MODEL_METADATA_IMPL_H_
