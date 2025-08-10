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
#ifndef MACHINA_LITE_NNAPI_SL_INCLUDE_SUPPORT_LIBRARY_SYMBOLS_H_
#define MACHINA_LITE_NNAPI_SL_INCLUDE_SUPPORT_LIBRARY_SYMBOLS_H_

#include <stddef.h>
// Changed when importing from AOSP
#include "machina/lite/nnapi/sl/public/NeuralNetworksSupportLibraryImpl.h"

// If you are linking against SL driver implementation through DT_NEEDED,
// you can use this declaration to access its implementation instead
// of doing dlsym.

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Get the NNAPI SL Driver NnApiSLDriverImpl with all
 * driver functions.
 */
NnApiSLDriverImpl* ANeuralNetworks_getSLDriverImpl();

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // MACHINA_LITE_NNAPI_SL_INCLUDE_SUPPORT_LIBRARY_SYMBOLS_H_
