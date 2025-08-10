/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 26, 2025.
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

// Functions for getting information about kernels registered in the binary.
#ifndef MACHINA_PYTHON_UTIL_KERNEL_REGISTRY_H_
#define MACHINA_PYTHON_UTIL_KERNEL_REGISTRY_H_

#include "machina/core/framework/graph.pb.h"
#include "machina/core/platform/types.h"

namespace machina {
namespace swig {

// Returns the kernel class name required to execute <node_def> on the device
// type of <node_def.device>, or an empty string if the kernel class is not
// found or the device name is invalid.
string TryFindKernelClass(const string& serialized_node_def);

}  // namespace swig
}  // namespace machina

#endif  // MACHINA_PYTHON_UTIL_KERNEL_REGISTRY_H_
