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

#ifndef MACHINA_LITE_KERNELS_GRU_CELL_H_
#define MACHINA_LITE_KERNELS_GRU_CELL_H_

#include "machina/lite/kernels/cpu_backend_context.h"
#include "machina/lite/kernels/internal/tensor.h"

namespace tflite {
namespace ops {
namespace custom {
namespace gru_cell {

void GruCell(const RuntimeShape& input_shape, const float* input,
             const RuntimeShape& state_shape, const float* input_state,
             const RuntimeShape& gate_weight_shape, const float* gate_weight,
             const RuntimeShape& gate_bias_shape, const float* gate_bias,
             const RuntimeShape& candidate_weight_shape,
             const float* candidate_weight,
             const RuntimeShape& candidate_bias_shape,
             const float* candidate_bias, const RuntimeShape& output_shape,
             float* output, float* output_state,
             const RuntimeShape& activation_shape, float* activation,
             const RuntimeShape& concat_shape, float* concat,
             const tflite::FullyConnectedParams& fc_params,
             tflite::CpuBackendContext* cpu_backend_context);

}  // namespace gru_cell
}  // namespace custom
}  // namespace ops
}  // namespace tflite

#endif  // MACHINA_LITE_KERNELS_GRU_CELL_H_
