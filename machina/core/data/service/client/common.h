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
#ifndef MACHINA_CORE_DATA_SERVICE_CLIENT_COMMON_H_
#define MACHINA_CORE_DATA_SERVICE_CLIENT_COMMON_H_

#include <cstdint>
#include <optional>
#include <string>

#include "absl/time/time.h"
#include "machina/core/data/service/common.pb.h"
#include "machina/core/protobuf/data_service.pb.h"

namespace machina {
namespace data {

// tf.data service parameters.
struct DataServiceParams final {
  std::string dataset_id;
  ProcessingModeDef processing_mode;
  std::string address;
  std::string protocol;
  std::string data_transfer_protocol;
  std::string job_name;
  int64_t repetition = 0;
  std::optional<int64_t> num_consumers;
  std::optional<int64_t> consumer_index;
  int64_t max_outstanding_requests = 0;
  absl::Duration task_refresh_interval;
  TargetWorkers target_workers = TargetWorkers::TARGET_WORKERS_UNSPECIFIED;
  DataServiceMetadata metadata;
  std::optional<CrossTrainerCacheOptions> cross_trainer_cache_options;
};

}  // namespace data
}  // namespace machina

#endif  // MACHINA_CORE_DATA_SERVICE_CLIENT_COMMON_H_
