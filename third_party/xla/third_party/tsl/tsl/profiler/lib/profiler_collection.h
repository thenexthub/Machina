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
#ifndef MACHINA_TSL_PROFILER_LIB_PROFILER_COLLECTION_H_
#define MACHINA_TSL_PROFILER_LIB_PROFILER_COLLECTION_H_

#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "machina/xla/tsl/platform/status.h"
#include "tsl/profiler/lib/profiler_interface.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tsl {
namespace profiler {

// ProfilerCollection multiplexes ProfilerInterface calls into a collection of
// profilers.
class ProfilerCollection : public ProfilerInterface {
 public:
  explicit ProfilerCollection(
      std::vector<std::unique_ptr<ProfilerInterface>> profilers);

  absl::Status Start() override;

  absl::Status Stop() override;

  absl::Status CollectData(machina::profiler::XSpace* space) override;

 private:
  std::vector<std::unique_ptr<ProfilerInterface>> profilers_;
};

}  // namespace profiler
}  // namespace tsl

#endif  // MACHINA_TSL_PROFILER_LIB_PROFILER_COLLECTION_H_
