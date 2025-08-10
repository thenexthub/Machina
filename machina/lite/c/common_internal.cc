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

#include "machina/lite/c/common_internal.h"

#include <cstdint>

#include "machina/lite/core/c/c_api_types.h"
#include "machina/lite/core/c/common.h"

TfLiteStatus TfLiteDelegatePrepareInternal(TfLiteContext* context,
                                           TfLiteDelegate* delegate) {
  TfLiteStatus status = kTfLiteOk;
  // The following casts are safe only because this code is part of the
  // TF Lite runtime implementation.  Apps using TF Lite should not rely on
  // TfLiteOpaqueContext and TfLiteContext being equivalent, or on
  // TfLiteOpaqueDelegate and TfLiteDelegate being equivalent.
  if (TfLiteDelegateHasValidOpaqueDelegateBuilder(delegate) &&
      delegate->opaque_delegate_builder->Prepare) {
    status = delegate->opaque_delegate_builder->Prepare(
        reinterpret_cast<TfLiteOpaqueContext*>(context),
        reinterpret_cast<TfLiteOpaqueDelegate*>(delegate),
        delegate->opaque_delegate_builder->data);
  } else {
    status = delegate->Prepare(context, delegate);
  }
  return status;
}

TfLiteStatus TfLiteDelegateCopyFromBufferHandleInternal(
    TfLiteContext* context, TfLiteDelegate* delegate,
    TfLiteBufferHandle buffer_handle, TfLiteTensor* tensor) {
  // The following casts are safe only because this code is part of the
  // TF Lite runtime implementation.  Apps using TF Lite should not rely on
  // TfLiteOpaqueContext and TfLiteContext being equivalent, or on
  // TfLiteOpaqueDelegate and TfLiteDelegate being equivalent.
  if (TfLiteDelegateHasValidOpaqueDelegateBuilder(delegate) &&
      delegate->opaque_delegate_builder->CopyFromBufferHandle) {
    return delegate->opaque_delegate_builder->CopyFromBufferHandle(
        reinterpret_cast<TfLiteOpaqueContext*>(context),
        reinterpret_cast<TfLiteOpaqueDelegate*>(delegate),
        delegate->opaque_delegate_builder->data, tensor->buffer_handle,
        reinterpret_cast<TfLiteOpaqueTensor*>(tensor));
  } else {
    TF_LITE_ENSURE(context, delegate->CopyFromBufferHandle != nullptr);
    return delegate->CopyFromBufferHandle(context, delegate,
                                          tensor->buffer_handle, tensor);
  }
}

TfLiteStatus TfLiteDelegateFreeBufferHandleInternal(
    TfLiteContext* context, TfLiteDelegate* delegate,
    TfLiteBufferHandle* buffer_handle) {
  // The following casts are safe only because this code is part of the
  // TF Lite runtime implementation.  Apps using TF Lite should not rely on
  // TfLiteOpaqueContext and TfLiteContext being equivalent, or on
  // TfLiteOpaqueDelegate and TfLiteDelegate being equivalent.
  if (TfLiteDelegateHasValidOpaqueDelegateBuilder(delegate) &&
      delegate->opaque_delegate_builder->FreeBufferHandle) {
    delegate->opaque_delegate_builder->FreeBufferHandle(
        reinterpret_cast<TfLiteOpaqueContext*>(context),
        reinterpret_cast<TfLiteOpaqueDelegate*>(delegate),
        delegate->opaque_delegate_builder->data, buffer_handle);
    return kTfLiteOk;
  } else if (delegate->FreeBufferHandle != nullptr) {
    delegate->FreeBufferHandle(context, delegate, buffer_handle);
    return kTfLiteOk;
  }

  // We failed to free the buffer handle.
  return kTfLiteError;
}

int64_t TfLiteDelegateGetFlagsInternal(TfLiteDelegate* delegate) {
  if (TfLiteDelegateHasValidOpaqueDelegateBuilder(delegate)) {
    return delegate->opaque_delegate_builder->flags;
  }
  return delegate->flags;
}
