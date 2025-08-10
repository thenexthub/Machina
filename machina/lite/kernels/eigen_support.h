/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 29, 2025.
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
#ifndef MACHINA_LITE_KERNELS_EIGEN_SUPPORT_H_
#define MACHINA_LITE_KERNELS_EIGEN_SUPPORT_H_

#include "machina/lite/core/c/common.h"

namespace EigenForTFLite {
struct ThreadPoolDevice;
}

namespace tflite {
namespace eigen_support {

// Let the framework know that the op will be using Eigen. If necessary a set of
// temporary Eigen objects might be created and placed in 'context'.
void IncrementUsageCounter(TfLiteContext* context);

// Let the framework know that the op stopped using Eigen. If there are no more
// usages all temporary Eigen objects will be deleted.
void DecrementUsageCounter(TfLiteContext* context);

// Fetch the ThreadPoolDevice associated with the provided context.
//
// Note: The caller must ensure that |IncrementUsageCounter()| has already been
// called. Moreover, it is *not* safe to cache the returned device; it may be
// invalidated if the context thread count changes.
const EigenForTFLite::ThreadPoolDevice* GetThreadPoolDevice(
    TfLiteContext* context);

}  // namespace eigen_support
}  // namespace tflite

#endif  // MACHINA_LITE_KERNELS_EIGEN_SUPPORT_H_
