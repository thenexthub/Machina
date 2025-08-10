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

#include <memory>
#include <utility>

#include "absl/status/statusor.h"
#include "machina/xla/pjrt/pjrt_client.h"
#include "machina/xla/pjrt/plugin/xla_cpu/cpu_client_options.h"
#include "machina/xla/pjrt/plugin/xla_cpu/xla_cpu_pjrt_client.h"
#include "machina/core/framework/types.h"
#include "machina/core/tfrt/common/pjrt_client_factory_options.h"
#include "machina/core/tfrt/common/pjrt_client_factory_registry.h"
#include "tsl/platform/statusor.h"

namespace xla {

absl::StatusOr<std::unique_ptr<xla::PjRtClient>> GetCpuClient(
    const PjrtClientFactoryOptions& option) {
  xla::CpuClientOptions cpu_options;
  cpu_options.asynchronous = option.cpu_options.asynchronous;

  TF_ASSIGN_OR_RETURN(std::unique_ptr<PjRtClient> client,
                      xla::GetXlaPjrtCpuClient(cpu_options));

  return std::move(client);
}

REGISTER_PJRT_CLIENT_FACTORY(cpu_client, machina::DEVICE_CPU, GetCpuClient);

}  // namespace xla
