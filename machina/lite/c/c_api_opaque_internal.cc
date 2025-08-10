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
#include "machina/lite/c/c_api_opaque_internal.h"

#include <memory>
#include <unordered_map>
#include <utility>

#include "machina/lite/core/api/op_resolver.h"
#include "machina/lite/core/c/common.h"
#include "machina/lite/core/c/operator.h"
#include "machina/lite/core/subgraph.h"

namespace tflite {
namespace internal {

namespace {

// Returns a dynamically allocated object; the caller is responsible for
// deallocating it using TfLiteOperatorDelete.
TfLiteOperator* MakeOperator(const TfLiteRegistration* registration,
                             int node_index) {
  // We need to allocate a new TfLiteOperator object and then
  // populate its state correctly, based on the contents in 'registration'.

  auto* registration_external = TfLiteOperatorCreate(
      static_cast<TfLiteBuiltinOperator>(registration->builtin_code),
      registration->custom_name, registration->version,
      /*user_data=*/nullptr);

  registration_external->node_index = node_index;

  return registration_external;
}

}  // anonymous namespace

TfLiteOperator* CommonOpaqueConversionUtil::CachedObtainOperator(
    OperatorsCache* registration_externals_cache,
    const TfLiteRegistration* registration, int node_index) {
  OpResolver::OpId op_id{registration->builtin_code, registration->custom_name,
                         registration->version};
  auto it = registration_externals_cache->find(op_id);
  if (it != registration_externals_cache->end()) {
    return it->second.get();
  }
  auto* registration_external = MakeOperator(registration, node_index);
  registration_externals_cache->insert(
      it, std::make_pair(op_id, registration_external));

  return registration_external;
}

TfLiteOperator* CommonOpaqueConversionUtil::ObtainOperator(
    TfLiteContext* context, const TfLiteRegistration* registration,
    int node_index) {
  auto* subgraph = static_cast<tflite::Subgraph*>(context->impl_);
  if (!subgraph->registration_externals_) {
    subgraph->registration_externals_ = std::make_shared<OperatorsCache>();
  }
  return CachedObtainOperator(subgraph->registration_externals_.get(),
                              registration, node_index);
}

}  // namespace internal
}  // namespace tflite
