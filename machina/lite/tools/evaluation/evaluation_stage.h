/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, May 15, 2025.
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
#ifndef MACHINA_LITE_TOOLS_EVALUATION_EVALUATION_STAGE_H_
#define MACHINA_LITE_TOOLS_EVALUATION_EVALUATION_STAGE_H_

#include "machina/lite/core/c/common.h"
#include "machina/lite/tools/evaluation/proto/evaluation_config.pb.h"

namespace tflite {
namespace evaluation {

// Superclass for a single stage of an EvaluationPipeline.
// Defines basic skeleton for sub-classes to implement.
//
// Ideally EvaluationStages should obtain access to initializer/input objects
// via Get/Set methods on pointers, and not take ownership unless necessary.
class EvaluationStage {
 public:
  // Initializes an EvaluationStage, including verifying the
  // EvaluationStageConfig. Returns kTfLiteError if initialization failed,
  // kTfLiteOk otherwise.
  //
  // Sub-classes are responsible for ensuring required class members are defined
  // via Get/Set methods.
  virtual TfLiteStatus Init() = 0;

  // An individual run of the EvaluationStage. This is where the task to be
  // evaluated takes place. Returns kTfLiteError if there was a failure,
  // kTfLiteOk otherwise.
  //
  // Sub-classes are responsible for ensuring they have access to required
  // inputs via Get/Set methods.
  virtual TfLiteStatus Run() = 0;

  // Returns the latest metrics based on all Run() calls made so far.
  virtual EvaluationStageMetrics LatestMetrics() = 0;

  virtual ~EvaluationStage() = default;

 protected:
  // Constructs an EvaluationStage.
  // Each subclass constructor must invoke this constructor.
  explicit EvaluationStage(const EvaluationStageConfig& config)
      : config_(config) {}

  EvaluationStageConfig config_;
};

}  // namespace evaluation
}  // namespace tflite

#endif  // MACHINA_LITE_TOOLS_EVALUATION_EVALUATION_STAGE_H_
