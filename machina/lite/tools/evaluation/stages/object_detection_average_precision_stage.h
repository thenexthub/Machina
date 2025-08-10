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
#ifndef MACHINA_LITE_TOOLS_EVALUATION_STAGES_OBJECT_DETECTION_AVERAGE_PRECISION_STAGE_H_
#define MACHINA_LITE_TOOLS_EVALUATION_STAGES_OBJECT_DETECTION_AVERAGE_PRECISION_STAGE_H_

#include <vector>

#include "machina/lite/c/c_api_types.h"
#include "machina/lite/tools/evaluation/evaluation_stage.h"
#include "machina/lite/tools/evaluation/proto/evaluation_config.pb.h"
#include "machina/lite/tools/evaluation/proto/evaluation_stages.pb.h"
#include "machina/lite/tools/evaluation/stages/utils/image_metrics.h"

namespace tflite {
namespace evaluation {

// EvaluationStage to compute Average Precision for Object Detection Task.
// Computes Average Precision per-IoU threshold (averaged across all classes),
// and then mean Average Precision (mAP) as the average AP value across all
// thresholds.
class ObjectDetectionAveragePrecisionStage : public EvaluationStage {
 public:
  explicit ObjectDetectionAveragePrecisionStage(
      const EvaluationStageConfig& config)
      : EvaluationStage(config) {}

  TfLiteStatus Init() override;

  TfLiteStatus Run() override;

  EvaluationStageMetrics LatestMetrics() override;

  // Call before Run().
  void SetEvalInputs(const ObjectDetectionResult& predicted_objects,
                     const ObjectDetectionResult& ground_truth_objects) {
    predicted_objects_ = predicted_objects;
    ground_truth_objects_ = ground_truth_objects;
  }

 private:
  int num_classes_ = -1;
  ObjectDetectionResult predicted_objects_;
  ObjectDetectionResult ground_truth_objects_;
  int current_image_index_ = 0;

  // One inner vector per class for ground truth objects.
  std::vector<std::vector<image::Detection>> ground_truth_object_vectors_;
  // One inner vector per class for predicted objects.
  std::vector<std::vector<image::Detection>> predicted_object_vectors_;
};

}  // namespace evaluation
}  // namespace tflite

#endif  // MACHINA_LITE_TOOLS_EVALUATION_STAGES_OBJECT_DETECTION_AVERAGE_PRECISION_STAGE_H_
