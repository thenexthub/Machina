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
#include "machina/lite/kernels/variants/register_list_ops.h"

#include "machina/lite/kernels/variants/list_ops_lib.h"
#include "machina/lite/mutable_op_resolver.h"

namespace tflite {
namespace variants {
namespace ops {

void RegisterListOps(MutableOpResolver* resolver) {
  resolver->AddCustom("TensorListReserve", Register_LIST_RESERVE());
  resolver->AddCustom("TensorListStack", Register_LIST_STACK());
  resolver->AddCustom("TensorListSetItem", Register_LIST_SET_ITEM());
  resolver->AddCustom("TensorListFromTensor", Register_LIST_FROM_TENSOR());
  resolver->AddCustom("TensorListGetItem", Register_LIST_GET_ITEM());
  resolver->AddCustom("TensorListLength", Register_LIST_LENGTH());
  resolver->AddCustom("TensorListElementShape", Register_LIST_ELEMENT_SHAPE());
  resolver->AddCustom("TensorListPopBack", Register_LIST_POP_BACK());
  resolver->AddCustom("TensorListPushBack", Register_LIST_PUSH_BACK());
  resolver->AddCustom("VariantAddN", Register_VARIANT_ADD_N());
  resolver->AddCustom("VariantZerosLike", Register_VARIANT_ZEROS_LIKE());
}

}  // namespace ops
}  // namespace variants
}  // namespace tflite

void RegisterSelectedOps(tflite::MutableOpResolver* resolver) {
  tflite::variants::ops::RegisterListOps(resolver);
}
