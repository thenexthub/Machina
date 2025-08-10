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

#include <vector>

#include "machina/c/eager/abstract_context.h"
#include "machina/c/eager/abstract_tensor_handle.h"
#include "machina/c/eager/c_api.h"
#include "machina/c/eager/c_api_unified_experimental.h"
#include "machina/c/eager/c_api_unified_experimental_internal.h"
#include "machina/c/eager/immediate_execution_context.h"
#include "machina/c/eager/immediate_execution_tensor_handle.h"
#include "machina/c/eager/tfe_context_internal.h"
#include "machina/c/eager/tfe_tensorhandle_internal.h"
#include "machina/c/tf_status.h"
#include "machina/core/lib/llvm_rtti/llvm_rtti.h"
#include "machina/core/platform/strcat.h"

// =============================================================================
// Public C API entry points
// These are only the entry points specific to the Eager API.
// =============================================================================

using machina::AbstractContext;
using machina::AbstractTensorHandle;
using machina::dyn_cast;
using machina::ImmediateExecutionContext;
using machina::ImmediateExecutionTensorHandle;
using machina::string;
using machina::unwrap;
using machina::wrap;
using machina::strings::StrCat;

TF_ExecutionContext* TF_NewEagerExecutionContext(TFE_ContextOptions* options,
                                                 TF_Status* s) {
  TFE_Context* c_ctx = TFE_NewContext(options, s);
  if (TF_GetCode(s) != TF_OK) {
    return nullptr;
  }
  return wrap(static_cast<AbstractContext*>(unwrap(c_ctx)));
}

TF_AbstractTensor* TF_CreateAbstractTensorFromEagerTensor(TFE_TensorHandle* t,
                                                          TF_Status* s) {
  return wrap(static_cast<AbstractTensorHandle*>(unwrap(t)));
}

TFE_TensorHandle* TF_AbstractTensorGetEagerTensor(TF_AbstractTensor* at,
                                                  TF_Status* s) {
  auto handle = dyn_cast<ImmediateExecutionTensorHandle>(unwrap(at));
  if (!handle) {
    string msg =
        StrCat("Not an eager tensor handle.", reinterpret_cast<uintptr_t>(at));
    TF_SetStatus(s, TF_INVALID_ARGUMENT, msg.c_str());
    return nullptr;
  }
  return wrap(handle);
}

TFE_Context* TF_ExecutionContextGetTFEContext(TF_ExecutionContext* ctx,
                                              TF_Status* s) {
  auto imm_ctx = dyn_cast<ImmediateExecutionContext>(unwrap(ctx));
  if (!imm_ctx) {
    string msg =
        StrCat("Not an eager context.", reinterpret_cast<uintptr_t>(ctx));
    TF_SetStatus(s, TF_INVALID_ARGUMENT, msg.c_str());
    return nullptr;
  }
  return wrap(imm_ctx);
}
