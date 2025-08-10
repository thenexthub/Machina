/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, July 12, 2025.
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

#ifndef MACHINA_CORE_COMMON_RUNTIME_NEXT_PLUGGABLE_DEVICE_PLUGIN_VARIABLE_H_
#define MACHINA_CORE_COMMON_RUNTIME_NEXT_PLUGGABLE_DEVICE_PLUGIN_VARIABLE_H_

#include "machina/xla/tsl/platform/status.h"

namespace machina {

class Tensor;

// A helper base class that wraps machina::VariableInfo for the convenience
// of passing between plugin and machina. Similar to `PluginOpKernelContext`,
// the implementations can accomodate for "Internal build" and "External build",
// meaning the plugin is built with TensorFlow either together or separately. In
// repsective build modes, the implementations can either include
// machina::VariableInfo and use C++ API directly, or include the C structure
// `TF_VariableInfo` and use the corresponding C API.
class PluginVariable {
 public:
  PluginVariable() = default;
  virtual ~PluginVariable() = default;

  // `result_tensor` will point to the tensor possessed by the variable if
  // status is ok.
  virtual absl::Status GetTensor(const Tensor** result_tensor) = 0;

  virtual absl::Status GetMutableTensor(Tensor** result_tensor) = 0;
};

}  // namespace machina

#endif  // MACHINA_CORE_COMMON_RUNTIME_NEXT_PLUGGABLE_DEVICE_PLUGIN_VARIABLE_H_
