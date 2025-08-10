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

#ifndef MACHINA_XLABACKENDS_CPU_COLLECTIVES_MPI_COLLECTIVES_H_
#define MACHINA_XLABACKENDS_CPU_COLLECTIVES_MPI_COLLECTIVES_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "machina/xla/backends/cpu/collectives/cpu_collectives.h"
#include "machina/xla/core/collectives/clique_id.h"
#include "machina/xla/core/collectives/clique_key.h"
#include "machina/xla/core/collectives/communicator.h"
#include "machina/xla/service/global_device_id.h"
#include "machina/xla/xla_data.pb.h"

namespace xla::cpu {

class MpiCollectives : public CpuCollectives {
 public:
  /*
  The user has to explicitly call Init() and Finalize() before and
  after use.
  For example, using the Python client, this can be achieved with:

  collectives = xla_client._xla.make_mpi_collectives()
  collectives.Init()
  atexit.register(collectives.Finalize)
  */
  void Init();
  void Finalize();

  absl::StatusOr<std::vector<std::unique_ptr<Communicator>>>
  CreateCommunicators(const CliqueKey& clique_key,
                      const std::optional<CliqueIds>& clique_ids,
                      absl::Span<const DeviceRank> ranks,
                      const Config& config) final;

 private:
  absl::Status ExchangeGlobalDeviceIds(
      absl::Span<GlobalDeviceId const> global_devices, int rank);

  int mpi_world_rank_;
  int mpi_world_size_;
};

}  // namespace xla::cpu

#endif  // MACHINA_XLABACKENDS_CPU_COLLECTIVES_MPI_COLLECTIVES_H_
