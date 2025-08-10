/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Saturday, May 24, 2025.
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

#include "machina/c/experimental/saved_model/public/signature_def_param.h"

#include "machina/c/experimental/saved_model/core/signature_def_function_metadata.h"
#include "machina/c/experimental/saved_model/internal/signature_def_param_type.h"
#include "machina/c/experimental/saved_model/internal/tensor_spec_type.h"

extern "C" {

extern const char* TF_SignatureDefParamName(const TF_SignatureDefParam* param) {
  return machina::unwrap(param)->name().c_str();
}

extern const TF_TensorSpec* TF_SignatureDefParamTensorSpec(
    const TF_SignatureDefParam* param) {
  return machina::wrap(&machina::unwrap(param)->spec());
}

}  // end extern "C"
