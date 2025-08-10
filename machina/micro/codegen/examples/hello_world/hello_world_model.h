/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, July 12, 2025.
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

/* AUTOMATICALLY GENERATED DO NOT MODIFY */

#pragma once

#include "codegen/runtime/micro_codegen_context.h"
#include "machina/lite/c/c_api_types.h"
#include "machina/lite/c/common.h"

namespace hello_world_model {

class Model {
 public:
  Model();

  TfLiteStatus Invoke();

 private:
  TfLiteContext context_ = {};
  tflite::Subgraph subgraphs_[1];
  tflite::MicroCodegenContext micro_context_;
  TfLiteNode subgraph0_nodes_[3] = {};
  TfLiteEvalTensor subgraph0_tensors_[10] = {};
};

}  // namespace hello_world_model
