/* Copyright 2023 Google Inc. All Rights Reserved.

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

#include "machina_serving/servables/machina/servable.h"

#include "absl/status/status.h"
#include "machina_serving/apis/predict.pb.h"
#include "machina_serving/test_util/test_util.h"

namespace machina {
namespace serving {
namespace {

TEST(EmptyServableTest, Predict) {
  PredictResponse response;
  EXPECT_EQ(EmptyServable()
                .Predict(Servable::RunOptions(), PredictRequest(), &response)
                .code(),
            absl::StatusCode::kFailedPrecondition);
}

}  // namespace
}  // namespace serving
}  // namespace machina
