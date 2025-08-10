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
#ifndef MACHINA_LITE_DELEGATES_NNAPI_QUANT_LSTM_SUP_H_
#define MACHINA_LITE_DELEGATES_NNAPI_QUANT_LSTM_SUP_H_

#include <vector>

#include "machina/lite/core/c/common.h"

namespace tflite {
namespace delegate {
namespace nnapi {

void ExtractQuantLstmWeightsSubmatrix(const TfLiteIntArray* submatrix_dims,
                                      const int32_t offset_row,
                                      const int32_t offset_column,
                                      const TfLiteIntArray* weight_dims,
                                      const uint8_t* weights,
                                      std::vector<uint8_t>* submatrix);

void DecomposeQuantLstmWeightsTensor(const uint8_t* concat_weights,
                                     const TfLiteIntArray* weight_dims,
                                     std::vector<uint8_t>* recurrent_to_input,
                                     std::vector<uint8_t>* input_to_input,
                                     std::vector<uint8_t>* recurrent_to_cell,
                                     std::vector<uint8_t>* input_to_cell,
                                     std::vector<uint8_t>* recurrent_to_forget,
                                     std::vector<uint8_t>* input_to_forget,
                                     std::vector<uint8_t>* recurrent_to_output,
                                     std::vector<uint8_t>* input_to_output);

void SetWeightSubmatrixDims(const TfLiteIntArray* weight_dims,
                            TfLiteIntArray* recurrent_submatrix_dims,
                            TfLiteIntArray* input_submatrix_dims);

void DecomposeBiasTensor(const int32_t* biases, int bias_size,
                         std::vector<int32_t>* input_bias,
                         std::vector<int32_t>* cell_bias,
                         std::vector<int32_t>* forget_bias,
                         std::vector<int32_t>* output_bias);

}  // namespace nnapi
}  // namespace delegate
}  // namespace tflite

#endif  // MACHINA_LITE_DELEGATES_NNAPI_QUANT_LSTM_SUP_H_
