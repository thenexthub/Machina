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
#ifndef MACHINA_LITE_TOCO_LOGGING_CONVERSION_LOG_UTIL_H_
#define MACHINA_LITE_TOCO_LOGGING_CONVERSION_LOG_UTIL_H_

#include <map>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "machina/lite/toco/logging/toco_conversion_log.pb.h"
#include "machina/lite/toco/model.h"

namespace toco {

// This function scans through the error message string, extracts the part about
// missing ops and prunes away all other information in the error info.
std::string SanitizeErrorMessage(absl::string_view error_message);

// Populates the TocoConversionLog proto after analyzing the model.
void PopulateConversionLog(const Model& model, TocoConversionLog* log);

// Returns the names of the operators in the model.
std::vector<std::string> GetOperatorNames(const Model& model);

// Counts the number of different types of operators in the model:
// Built-in ops, custom ops and select ops.
// Each map is mapping from the name of the operator (such as 'Conv') to its
// total number of occurrences in the model.
void CountOperatorsByType(const Model& model,
                          std::map<std::string, int>* built_in_ops,
                          std::map<std::string, int>* custom_ops,
                          std::map<std::string, int>* select_ops);

// Gets the input and output types of the model. The input and output is
// specified by model.flags.input_arrays and model.flags.output_arrays.
void GetInputAndOutputTypes(
    const Model& model,
    TFLITE_PROTO_NS::RepeatedPtrField<std::string>* input_types,
    TFLITE_PROTO_NS::RepeatedPtrField<std::string>* output_types);

// Calculates signatures for all the ops in the model. An op signature is
// defined by its input/output shapes and types, op name and its version.
void GetOpSignatures(
    const Model& model,
    TFLITE_PROTO_NS::RepeatedPtrField<std::string>* op_signatures);

// TODO(b/123519920): Implement this.
// Calculates a unique hash for the model.
std::string GetModelHash(const Model& model);

}  // namespace toco

#endif  // MACHINA_LITE_TOCO_LOGGING_CONVERSION_LOG_UTIL_H_
