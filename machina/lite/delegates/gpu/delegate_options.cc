/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, June 15, 2025.
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

#include "machina/lite/delegates/gpu/delegate_options.h"

#include <limits>

TfLiteGpuDelegateOptionsV2 TfLiteGpuDelegateOptionsV2Default() {
  TfLiteGpuDelegateOptionsV2 options;
  // set it to -1 to detect whether it was later adjusted.
  options.is_precision_loss_allowed = -1;
  options.inference_preference =
      TFLITE_GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER;
  options.inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MAX_PRECISION;
  options.inference_priority2 = TFLITE_GPU_INFERENCE_PRIORITY_AUTO;
  options.inference_priority3 = TFLITE_GPU_INFERENCE_PRIORITY_AUTO;
  options.experimental_flags = TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_QUANT;
  options.max_delegated_partitions = 1;
  options.model_token = nullptr;
  options.serialization_dir = nullptr;
#ifdef TFLITE_DEBUG_DELEGATE
  options.first_delegate_node_index = 0;
  options.last_delegate_node_index = std::numeric_limits<int>::max();
#endif
#ifdef TFLITE_GPU_ENABLE_INVOKE_LOOP
  options.gpu_invoke_loop_times = 1;
#endif
  return options;
}
