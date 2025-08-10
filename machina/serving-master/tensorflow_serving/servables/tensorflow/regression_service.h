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

#ifndef MACHINA_SERVING_SERVABLES_MACHINA_REGRESSION_SERVICE_H_
#define MACHINA_SERVING_SERVABLES_MACHINA_REGRESSION_SERVICE_H_

#include "machina/core/lib/core/status.h"
#include "machina/core/protobuf/config.pb.h"
#include "machina_serving/apis/regression.pb.h"
#include "machina_serving/model_servers/server_core.h"

namespace machina {
namespace serving {

// Utility methods for implementation of
// machina_serving/apis/regression-service.proto.
class TensorflowRegressionServiceImpl final {
 public:
  static Status Regress(const RunOptions& run_options, ServerCore* core,
                        const thread::ThreadPoolOptions& thread_pool_options,
                        const RegressionRequest& request,
                        RegressionResponse* response);

  // Like Regress(), but uses 'model_spec' instead of the one embedded in
  // 'request'.
  static Status RegressWithModelSpec(
      const RunOptions& run_options, ServerCore* core,
      const thread::ThreadPoolOptions& thread_pool_options,
      const ModelSpec& model_spec, const RegressionRequest& request,
      RegressionResponse* response);
};

}  // namespace serving
}  // namespace machina

#endif  // MACHINA_SERVING_SERVABLES_MACHINA_REGRESSION_SERVICE_H_
