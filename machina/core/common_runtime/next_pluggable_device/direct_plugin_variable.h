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

#ifndef MACHINA_CORE_COMMON_RUNTIME_NEXT_PLUGGABLE_DEVICE_DIRECT_PLUGIN_VARIABLE_H_
#define MACHINA_CORE_COMMON_RUNTIME_NEXT_PLUGGABLE_DEVICE_DIRECT_PLUGIN_VARIABLE_H_

#include <string>

#include "absl/status/status.h"
#include "machina/compiler/jit/variable_info.h"
#include "machina/xla/tsl/platform/status.h"
#include "machina/core/common_runtime/next_pluggable_device/plugin_variable.h"

namespace machina {

class DirectPluginOpKernelContext;

class DirectPluginVariable : public PluginVariable {
 public:
  DirectPluginVariable(int index, const std::string& name, Var* var);
  absl::Status GetTensor(const Tensor** result_tensor) override {
    *result_tensor = var_info_.var()->tensor();
    return absl::OkStatus();
  }

  absl::Status GetMutableTensor(Tensor** result_tensor) override {
    *result_tensor = var_info_.var()->tensor();
    return absl::OkStatus();
  }

  VariableInfo* GetVariableInfo() { return &var_info_; }

  friend DirectPluginOpKernelContext;

 private:
  VariableInfo var_info_{0, "", nullptr};
};

}  // namespace machina

#endif  // MACHINA_CORE_COMMON_RUNTIME_NEXT_PLUGGABLE_DEVICE_DIRECT_PLUGIN_VARIABLE_H_
