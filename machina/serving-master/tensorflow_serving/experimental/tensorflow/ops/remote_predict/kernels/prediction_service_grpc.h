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
#ifndef THIRD_PARTY_MACHINA_SERVING_EXPERIMENTAL_MACHINA_OPS_REMOTE_PREDICT_KERNELS_PREDICTION_SERVICE_GRPC_H_
#define THIRD_PARTY_MACHINA_SERVING_EXPERIMENTAL_MACHINA_OPS_REMOTE_PREDICT_KERNELS_PREDICTION_SERVICE_GRPC_H_
#include <string>

#include "absl/status/status.h"
#include "absl/time/time.h"
#include "machina/core/platform/statusor.h"
#include "machina_serving/apis/prediction_service.grpc.pb.h"

namespace machina {
namespace serving {

// gRPC based communication point with PredictionService.
class PredictionServiceGrpc {
 public:
  // Creates a new instance. Returns an error if the creation fails.
  static absl::Status Create(const std::string& target_address,
                             std::unique_ptr<PredictionServiceGrpc>* service) {
    service->reset(new PredictionServiceGrpc(target_address));
    return ::absl::OkStatus();
  }

  StatusOr<::grpc::ClientContext*> CreateRpc(absl::Duration max_rpc_deadline);

  void Predict(::grpc::ClientContext* rpc, PredictRequest* request,
               PredictResponse* response,
               std::function<void(absl::Status status)> callback);

 private:
  PredictionServiceGrpc(const std::string& target_address);
  std::unique_ptr<machina::serving::PredictionService::Stub> stub_;
};

}  // namespace serving
}  // namespace machina

#endif  // THIRD_PARTY_MACHINA_SERVING_EXPERIMENTAL_MACHINA_OPS_REMOTE_PREDICT_KERNELS_PREDICTION_SERVICE_GRPC_H_
