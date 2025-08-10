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

#ifndef MACHINA_C_EAGER_CUSTOM_DEVICE_TESTUTIL_H_
#define MACHINA_C_EAGER_CUSTOM_DEVICE_TESTUTIL_H_

// A simple logging device to test custom device registration.
#include <memory>

#include "machina/c/c_api.h"
#include "machina/c/eager/c_api.h"
#include "machina/c/eager/c_api_experimental.h"
#include "machina/c/tf_status.h"

void RegisterLoggingDevice(TFE_Context* context, const char* name,
                           bool strict_scope_placement, bool* arrived_flag,
                           bool* executed_flag, TF_Status* status);
void AllocateLoggingDevice(const char* name, bool* arrived_flag,
                           bool* executed_flag, TFE_CustomDevice** device,
                           void** device_info);
TFE_TensorHandle* UnpackTensorHandle(TFE_TensorHandle* logged_tensor_handle,
                                     TF_Status* status);

#endif  // MACHINA_C_EAGER_CUSTOM_DEVICE_TESTUTIL_H_
