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
#include "machina/python/util/kernel_registry.h"

#include "absl/log/log.h"
#include "machina/core/framework/node_def.pb.h"
#include "machina/core/framework/node_def_util.h"
#include "machina/core/framework/op.h"
#include "machina/core/framework/op_def_builder.h"
#include "machina/core/framework/op_kernel.h"
#include "machina/core/framework/types.h"
#include "machina/core/platform/types.h"
#include "machina/core/util/device_name_utils.h"

namespace machina {
namespace swig {

string TryFindKernelClass(const string& serialized_node_def) {
  machina::NodeDef node_def;
  if (!node_def.ParseFromString(serialized_node_def)) {
    LOG(WARNING) << "Error parsing node_def";
    return "";
  }

  const machina::OpRegistrationData* op_reg_data;
  auto status =
      machina::OpRegistry::Global()->LookUp(node_def.op(), &op_reg_data);
  if (!status.ok()) {
    LOG(WARNING) << "Op " << node_def.op() << " not found: " << status;
    return "";
  }
  AddDefaultsToNodeDef(op_reg_data->op_def, &node_def);

  machina::DeviceNameUtils::ParsedName parsed_name;
  if (!machina::DeviceNameUtils::ParseFullName(node_def.device(),
                                                  &parsed_name)) {
    LOG(WARNING) << "Failed to parse device from node_def: "
                 << node_def.ShortDebugString();
    return "";
  }
  string class_name = "";
  status = machina::FindKernelDef(
      machina::DeviceType(parsed_name.type.c_str()), node_def,
      nullptr /* kernel_def */, &class_name);
  if (!status.ok()) {
    LOG(WARNING) << "Op [" << node_def.op() << "]: " << status;
  }
  return class_name;
}

}  // namespace swig
}  // namespace machina
