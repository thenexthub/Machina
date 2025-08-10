/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 29, 2025.
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
#ifndef MACHINA_LITE_TOOLS_OPTIMIZE_QUANTIZE_MODEL_H_
#define MACHINA_LITE_TOOLS_OPTIMIZE_QUANTIZE_MODEL_H_

#include <memory>
#include <unordered_set>

#include "machina/lite/context.h"
#include "machina/lite/core/api/error_reporter.h"
#include "machina/lite/core/model.h"
#include "machina/lite/schema/schema_generated.h"
#include "machina/lite/util.h"

namespace tflite {
namespace optimize {

// Quantizes input_model and populates the provided builder with the new model.
// input_model is required to have min/max information populated in its
// quantization params.
//
// Inputs and output types default to float instead of a quantized type.
//
// Note: This is a private API, subject to change.
TfLiteStatus QuantizeModel(flatbuffers::FlatBufferBuilder* builder,
                           ModelT* input_model, ErrorReporter* error_reporter);

// Same as above, but the types of quantized inputs and outputs are
// configurable.
//
// Note: This is a private API, subject to change.
TfLiteStatus QuantizeModel(flatbuffers::FlatBufferBuilder* builder,
                           ModelT* input_model, const TensorType& input_type,
                           const TensorType& output_type,
                           ErrorReporter* error_reporter);

// Same as above, but can enable allowing float intermediate operations for ops
// that do not yet support quantizable.
//
// Note: This is a private API, subject to change.
TfLiteStatus QuantizeModel(flatbuffers::FlatBufferBuilder* builder,
                           ModelT* input_model, const TensorType& input_type,
                           const TensorType& output_type, bool allow_float,
                           ErrorReporter* error_reporter);

// Same as above but with added option of disabling per channel quantization
//
// Note: This is a private API, subject to change.
TfLiteStatus QuantizeModel(
    flatbuffers::FlatBufferBuilder* builder, ModelT* input_model,
    const TensorType& input_type, const TensorType& output_type,
    bool allow_float, bool disable_per_channel,
    bool disable_per_channel_quantization_for_dense_layers,
    ErrorReporter* error_reporter);

// Same as above but with added option of handling quantization of external
// state tensors. This assumes first input and output tensors are ouputs and
// rest are state tensors which are quantized later with type as
// activation type (hence no fake quant ops).
// Note: This is a private API, subject to change.
TfLiteStatus QuantizeModel(
    flatbuffers::FlatBufferBuilder* builder, ModelT* input_model,
    const TensorType& input_type, const TensorType& output_type,
    bool allow_float, bool disable_per_channel,
    bool disable_per_channel_quantization_for_dense_layers,
    ErrorReporter* error_reporter, bool handle_external_state);

// Same as above, but enables only quantizing an allowlist of operations,
// specified by their operator output name.
//
// Note: This is a private API, subject to change.
TfLiteStatus QuantizeModel(flatbuffers::FlatBufferBuilder* builder,
                           ModelT* input_model, const TensorType& input_type,
                           const TensorType& output_type, bool allow_float,
                           const std::unordered_set<string>& operator_names,
                           ErrorReporter* error_reporter);

// Same as above, but enables to provide activation type, which
// could be TensorType_INT16 or TensorType_INT8.
//
// Note: This is a private API, subject to change.
TfLiteStatus QuantizeModel(flatbuffers::FlatBufferBuilder* builder,
                           ModelT* model, const TensorType& input_type,
                           const TensorType& output_type, bool allow_float,
                           const std::unordered_set<string>& operator_names,
                           const TensorType& activations_type,
                           const TensorType& bias_type,
                           ErrorReporter* error_reporter);

// Same as above, but all operators supporting quantization are quantized.
//
// Note: This is a private API, subject to change.
TfLiteStatus QuantizeModelAllOperators(
    flatbuffers::FlatBufferBuilder* builder, ModelT* model,
    const TensorType& input_type, const TensorType& output_type,
    bool allow_float, const TensorType& activations_type,
    const TensorType& bias_type, ErrorReporter* error_reporter);

// Same as above, but allows disabling per channel quantization.
//
// Note: This is a private API, subject to change.
TfLiteStatus QuantizeModelAllOperators(
    flatbuffers::FlatBufferBuilder* builder, ModelT* model,
    const TensorType& input_type, const TensorType& output_type,
    bool allow_float, const TensorType& activations_type,
    const TensorType& bias_type, bool disable_per_channel,
    bool disable_per_channel_quantization_for_dense_layers,
    ErrorReporter* error_reporter);

// Quantizes input_model and populates the provided builder with the new model
// with all possible input parameters including disabling per_channel
// quantization.
//
// All functions above call this function underneath.
TfLiteStatus QuantizeModel(
    flatbuffers::FlatBufferBuilder* builder, ModelT* model,
    const TensorType& input_type, const TensorType& output_type,
    bool allow_float, const std::unordered_set<string>& operator_names,
    const TensorType& activations_type, const TensorType& bias_type,
    bool disable_per_channel,
    bool disable_per_channel_quantization_for_dense_layers,
    ErrorReporter* error_reporter, bool handle_external_state);

}  // namespace optimize
}  // namespace tflite

#endif  // MACHINA_LITE_TOOLS_OPTIMIZE_QUANTIZE_MODEL_H_
