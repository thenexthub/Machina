/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Friday, August 8, 2025.
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

#include "machina/compiler/mlir/lite/transforms/converter_pass_options_setter.h"

#include "machina/compiler/mlir/lite/transforms/optimize_broadcast_like_pass_options.h"
#include "machina/compiler/mlir/lite/transforms/optimize_pass_options.h"
#include "machina/compiler/mlir/lite/transforms/pass_options.h"
#include "machina/compiler/mlir/lite/transforms/variable_freezing_pipeline_options.h"

namespace mlir {
namespace TFL {

void ConverterPassOptionsSetter::SetOptions(
    OptimizePassOptions& options) const {
  options.enable_canonicalization = true;
  options.disable_fuse_mul_and_fc = converter_flags_.disable_fuse_mul_and_fc();
}

void ConverterPassOptionsSetter::SetOptions(
    VariableFreezingPipelineOptions& options) const {
  options.enable_tflite_variables = pass_config_.enable_tflite_variables;
}

void ConverterPassOptionsSetter::SetOptions(
    OptimizeBroadcastLikePassOptions& options) const {
  // options.unsafe_fuse_dynamic_shaped_broadcast =
  //     converter_flags_.unsafe_fuse_dynamic_shaped_broadcast();
}

void ConverterPassOptionsSetter::SetOptions(EmptyPassOptions& options) const {}

}  // namespace TFL
}  // namespace mlir
