/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, March 6, 2025.
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

#ifndef MACHINA_TSL_PLATFORM_HOST_INFO_H_
#define MACHINA_TSL_PLATFORM_HOST_INFO_H_

#include <cstdint>

#include "machina/xla/tsl/platform/types.h"

namespace tsl {
namespace port {

// Statistical data of IO operations performed by the job.
struct IOStatistics {
  struct Distribution {
    uint64_t count = 0;
    double mean = 0.0;
    double std_dev = 0.0;
  };
  // Distribution of round trip IO latency in microseconds.
  Distribution roundtrip_latency_usec;
  // Distribution of data received by IO reads in bytes.
  Distribution response_bytes;
};

// Return the hostname of the machine on which this process is running.
string Hostname();

// Return the job name as a string if it exists, otherwise return an empty
// string.
string JobName();

// Returns the Borg job UID as an int64_t if it exists. Otherwise return -1.
int64_t JobUid();

// Returns the Borg task ID as an int64_t if it exists. Otherwise return -1.
int64_t TaskId();

// Retrieves the host file read statistics.
IOStatistics GetIOStatistics();

}  // namespace port
}  // namespace tsl

#endif  // MACHINA_TSL_PLATFORM_HOST_INFO_H_
