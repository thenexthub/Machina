/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Saturday, June 21, 2025.
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

#include <stdint.h>
#include <string.h>

#include <string>
#include <utility>

#include "machina/lite/core/c/builtin_op_data.h"
#include "machina/lite/core/c/common.h"
#include "machina/lite/core/subgraph.h"
#include "machina/lite/experimental/resource/resource_variable.h"
#include "machina/lite/kernels/internal/tensor.h"
#include "machina/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace var_handle {
// Util struct with params that identifies the resource.
struct VarParams {
  int resource_id;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  const auto* var_params =
      reinterpret_cast<const TfLiteVarHandleParams*>(buffer);
  VarParams* params = new VarParams;
  auto* subgraph = reinterpret_cast<Subgraph*>(context->impl_);
  // Create a new entry if doesn't exist, return the existing one otherwise.
  auto it = subgraph->resource_ids().insert(std::make_pair(
      std::make_pair(
          std::string(var_params->container ? var_params->container : ""),
          std::string(var_params->shared_name ? var_params->shared_name : "")),
      static_cast<int>(subgraph->resource_ids().size())));
  params->resource_id = it.first->second;
  return params;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<VarParams*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));

  output->allocation_type = kTfLiteArenaRwPersistent;
  const int kBytesRequired = sizeof(int32_t);
  TfLiteTensorRealloc(kBytesRequired, output);
  output->bytes = kBytesRequired;

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* op_data = static_cast<VarParams*>(node->user_data);
  TF_LITE_ENSURE(context, op_data != nullptr);

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));

  memcpy(output->data.raw, reinterpret_cast<char*>(&op_data->resource_id),
         sizeof(op_data->resource_id));
  return kTfLiteOk;
}

}  // namespace var_handle

TfLiteRegistration* Register_VAR_HANDLE() {
  static TfLiteRegistration r = {var_handle::Init, var_handle::Free,
                                 var_handle::Prepare, var_handle::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
