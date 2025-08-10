/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, May 15, 2025.
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

// This file provides general C++ utility functions for interacting with NNAPI.

#ifndef MACHINA_LITE_NNAPI_NNAPI_UTIL_H_
#define MACHINA_LITE_NNAPI_NNAPI_UTIL_H_

#include <string>
#include <vector>

#include "machina/lite/nnapi/nnapi_implementation.h"

namespace tflite {
namespace nnapi {

// Return std::vector consisting of pointers to null-terminated device names.
// These names are guaranteed to be valid for the lifetime of the application.
std::vector<const char*> GetDeviceNamesList();
// An overload that uses a client-provided NnApi* structure to request available
// devices instead of the static one provided by NnApiImplementation().
// The names are guaranteed to be valid for the lifetime of the application.
std::vector<const char*> GetDeviceNamesList(const NnApi* nnapi);

// Return a string containing the names of all available devices.
// Will take the format: "DeviceA,DeviceB,DeviceC"
std::string GetStringDeviceNamesList();
// An overload that uses a client-provided NnApi* structure to request available
// devices instead of the static one provided by NnApiImplementation().
std::string GetStringDeviceNamesList(const NnApi* nnapi);

}  // namespace nnapi
}  // namespace tflite

#endif  // MACHINA_LITE_NNAPI_NNAPI_UTIL_H_
