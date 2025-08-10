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
#ifndef MACHINA_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_NNAPI_SL_FAKE_IMPL_H_
#define MACHINA_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_NNAPI_SL_FAKE_IMPL_H_

namespace tflite {
namespace acceleration {

// Initialize the shared file used to check if NNAPI SL has been called
// and count the number of API calls.
void InitNnApiSlInvocationStatus();

// Checks if any API call to the fake NNAPI SL has been done.
bool WasNnApiSlInvoked();

// Returns the number of calls to the fake NNAPI SL.
int CountNnApiSlApiCalls();

}  // namespace acceleration
}  // namespace tflite
#endif  // MACHINA_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_NNAPI_SL_FAKE_IMPL_H_
