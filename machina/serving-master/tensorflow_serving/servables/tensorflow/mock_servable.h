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

#ifndef MACHINA_SERVING_SERVABLES_MACHINA_MOCK_SERVABLE_H_
#define MACHINA_SERVING_SERVABLES_MACHINA_MOCK_SERVABLE_H_

#include <memory>

#include <gmock/gmock.h>
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "machina_serving/apis/classification.pb.h"
#include "machina_serving/apis/get_model_metadata.pb.h"
#include "machina_serving/apis/inference.pb.h"
#include "machina_serving/apis/predict.pb.h"
#include "machina_serving/apis/regression.pb.h"
#include "machina_serving/servables/machina/servable.h"

namespace machina {
namespace serving {

class MockPredictStreamedContext : public PredictStreamedContext {
 public:
  MOCK_METHOD(absl::Status, ProcessRequest, (const PredictRequest& request),
              (final));
  MOCK_METHOD(absl::Status, Close, (), (final));
  MOCK_METHOD(absl::Status, WaitResponses, (), (final));
};

// A mock of machina::serving::Servable.
class MockServable : public Servable {
 public:
  MockServable() : Servable("", 0) {}
  ~MockServable() override = default;

  MOCK_METHOD(absl::Status, Classify,
              (const machina::serving::Servable::RunOptions& run_options,
               const machina::serving::ClassificationRequest& request,
               machina::serving::ClassificationResponse* response),
              (final));
  MOCK_METHOD(absl::Status, Regress,
              (const machina::serving::Servable::RunOptions& run_options,
               const machina::serving::RegressionRequest& request,
               machina::serving::RegressionResponse* response),
              (final));
  MOCK_METHOD(absl::Status, Predict,
              (const machina::serving::Servable::RunOptions& run_options,
               const machina::serving::PredictRequest& request,
               machina::serving::PredictResponse* response),
              (final));
  MOCK_METHOD(absl::StatusOr<std::unique_ptr<PredictStreamedContext>>,
              PredictStreamed,
              (const machina::serving::Servable::RunOptions& run_options,
               absl::AnyInvocable<
                   void(absl::StatusOr<machina::serving::PredictResponse>)>
                   response_callback),
              (final));
  MOCK_METHOD(absl::Status, MultiInference,
              (const machina::serving::Servable::RunOptions& run_options,
               const machina::serving::MultiInferenceRequest& request,
               machina::serving::MultiInferenceResponse* response),
              (final));
  MOCK_METHOD(absl::Status, GetModelMetadata,
              (const machina::serving::GetModelMetadataRequest& request,
               machina::serving::GetModelMetadataResponse* response),
              (final));
  MOCK_METHOD(bool, SupportsPaging, (), (const, final));
  MOCK_METHOD(absl::Status, Suspend, (), (final));
  MOCK_METHOD(absl::Status, Resume, (), (final));
};

}  // namespace serving
}  // namespace machina

#endif  // MACHINA_SERVING_SERVABLES_MACHINA_MOCK_SERVABLE_H_
