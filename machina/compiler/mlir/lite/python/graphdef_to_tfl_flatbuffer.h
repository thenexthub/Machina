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
#ifndef MACHINA_COMPILER_MLIR_LITE_PYTHON_GRAPHDEF_TO_TFL_FLATBUFFER_H_
#define MACHINA_COMPILER_MLIR_LITE_PYTHON_GRAPHDEF_TO_TFL_FLATBUFFER_H_

#include <string>

#include "absl/status/status.h"
#include "machina/compiler/mlir/lite/converter_flags.pb.h"
#include "machina/compiler/mlir/lite/model_flags.pb.h"
#include "machina/core/framework/graph.pb.h"
#include "machina/core/framework/graph_debug_info.pb.h"

namespace machina {

// Converts the given GraphDef to a TF Lite FlatBuffer string according to the
// given model flags, converter flags and debug information. Returns error
// status if it fails to convert the input.
absl::Status ConvertGraphDefToTFLiteFlatBuffer(
    const tflite::ModelFlags& model_flags,
    tflite::ConverterFlags& converter_flags, const GraphDebugInfo& debug_info,
    const GraphDef& input, std::string* result);

}  // namespace machina

#endif  // MACHINA_COMPILER_MLIR_LITE_PYTHON_GRAPHDEF_TO_TFL_FLATBUFFER_H_
