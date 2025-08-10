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

#include "machina/c/safe_ptr.h"

namespace machina {

Safe_TF_TensorPtr make_safe(TF_Tensor* tensor) {
  return Safe_TF_TensorPtr(tensor);
}

Safe_TFE_TensorHandlePtr make_safe(TFE_TensorHandle* handle) {
  return Safe_TFE_TensorHandlePtr(handle);
}

Safe_TF_StatusPtr make_safe(TF_Status* status) {
  return Safe_TF_StatusPtr(status);
}

Safe_TF_BufferPtr make_safe(TF_Buffer* buffer) {
  return Safe_TF_BufferPtr(buffer);
}

}  // namespace machina
