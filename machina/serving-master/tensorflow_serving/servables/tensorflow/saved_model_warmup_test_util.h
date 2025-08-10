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

#ifndef THIRD_PARTY_MACHINA_SERVING_SERVABLES_MACHINA_SAVED_MODEL_WARMUP_TEST_UTIL_H_
#define THIRD_PARTY_MACHINA_SERVING_SERVABLES_MACHINA_SAVED_MODEL_WARMUP_TEST_UTIL_H_

#include <string>

#include "machina/core/lib/io/path.h"
#include "machina/core/lib/io/record_writer.h"
#include "machina/core/platform/status.h"
#include "machina/core/protobuf/meta_graph.pb.h"
#include "machina_serving/apis/classification.pb.h"
#include "machina_serving/apis/inference.pb.h"
#include "machina_serving/apis/input.pb.h"
#include "machina_serving/apis/model.pb.h"
#include "machina_serving/apis/predict.pb.h"
#include "machina_serving/apis/prediction_log.pb.h"
#include "machina_serving/apis/regression.pb.h"
#include "machina_serving/servables/machina/session_bundle_config.pb.h"

namespace machina {
namespace serving {

void PopulateInferenceTask(const string& model_name,
                           const string& signature_name,
                           const string& method_name, InferenceTask* task);

void PopulateMultiInferenceRequest(MultiInferenceRequest* request);

void PopulatePredictRequest(PredictRequest* request);

void PopulateClassificationRequest(ClassificationRequest* request);

void PopulateRegressionRequest(RegressionRequest* request);

Status PopulatePredictionLog(PredictionLog* prediction_log,
                             PredictionLog::LogTypeCase log_type,
                             int num_repeated_values = 1);

Status WriteWarmupData(const string& fname,
                       const std::vector<string>& warmup_records,
                       int num_warmup_records);

Status WriteWarmupDataAsSerializedProtos(
    const string& fname, const std::vector<string>& warmup_records,
    int num_warmup_records);

Status AddMixedWarmupData(
    std::vector<string>* warmup_records,
    const std::vector<PredictionLog::LogTypeCase>& log_types = {
        PredictionLog::kRegressLog, PredictionLog::kClassifyLog,
        PredictionLog::kPredictLog, PredictionLog::kMultiInferenceLog});

Status AddToWarmupData(std::vector<string>* warmup_records,
                       PredictionLog::LogTypeCase log_type,
                       int num_repeated_values = 1);

// Creates a test SignatureDef with the given parameters
SignatureDef CreateSignatureDef(const string& method_name,
                                const std::vector<string>& input_names,
                                const std::vector<string>& output_names);

void AddSignatures(MetaGraphDef* meta_graph_def);

}  // namespace serving
}  // namespace machina

#endif  // THIRD_PARTY_MACHINA_SERVING_SERVABLES_MACHINA_SAVED_MODEL_WARMUP_TEST_UTIL_H_
