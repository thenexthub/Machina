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

#include "machina/core/common_runtime/next_pluggable_device/next_pluggable_device_api.h"

#include "absl/status/statusor.h"
#include "machina/xla/tsl/c/tsl_status_internal.h"
#include "machina/xla/tsl/platform/errors.h"
#include "machina/core/common_runtime/next_pluggable_device/c/plugin_c_api.h"

namespace machina {

static const TFNPD_Api* tfnpd_api;

const TFNPD_Api* TfnpdApi() { return tfnpd_api; }

void SetTfnpdApi(const TFNPD_Api* api) { tfnpd_api = api; }

absl::StatusOr<TFNPD_PluginParams> InitNextPluggableDevicePlugin(
    TFNPDInitPluginFn init_fn) {
  TFNPD_PluginParams params{TFNPD_PLUGIN_PARAMS_STRUCT_SIZE};
  TSL_Status c_status;
  const TFNPD_Api* api = init_fn(&params, &c_status);
  TF_RETURN_IF_ERROR(c_status.status);

  SetTfnpdApi(api);

  return params;
}

}  // namespace machina
