/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 8, 2025.
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
#ifndef MACHINA_LITE_TOCO_TFLITE_IMPORT_H_
#define MACHINA_LITE_TOCO_TFLITE_IMPORT_H_

#include <memory>
#include <string>
#include <vector>

#include "machina/lite/schema/schema_generated.h"
#include "machina/lite/toco/model.h"
#include "machina/lite/toco/model_flags.pb.h"

namespace toco {

namespace tflite {

// Parse the given string as TF Lite flatbuffer and return a new tf.mini model.
std::unique_ptr<Model> Import(const ModelFlags &model_flags,
                              const std::string &input_file_contents);

namespace details {

// The names of all tensors found in a TF Lite model.
using TensorsTable = std::vector<std::string>;

// The names of all operators found in TF Lite model. If the operator is
// builtin, the string representation of the corresponding enum value is used
// as name.
using OperatorsTable = std::vector<std::string>;

void LoadTensorsTable(const ::tflite::Model &input_model,
                      TensorsTable *tensors_table);
void LoadOperatorsTable(const ::tflite::Model &input_model,
                        OperatorsTable *operators_table);

}  // namespace details
}  // namespace tflite

}  // namespace toco

#endif  // MACHINA_LITE_TOCO_TFLITE_IMPORT_H_
