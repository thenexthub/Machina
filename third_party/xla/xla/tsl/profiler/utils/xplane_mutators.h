/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Friday, August 8, 2025.
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

#ifndef MACHINA_XLATSL_PROFILER_UTILS_XPLANE_MUTATORS_H_
#define MACHINA_XLATSL_PROFILER_UTILS_XPLANE_MUTATORS_H_

#include <memory>
#include <vector>

#include "machina/xla/tsl/profiler/utils/xplane_builder.h"

namespace tsl {
namespace profiler {

/*
 * Subclass of this interface will perform different mutatation to the event.
 * Checking eligibilities of event mutation is not responsible of this class.
 */
class XplaneEventMutator {
 public:
  virtual ~XplaneEventMutator() = default;

  // Mutate event by event specified by the event_metadata.
  virtual void Mutate(XEventBuilder& builder) = 0;
  // Mutate line by line if event_metadata() return nullptr.
  virtual void MutateEventsInLine(XLineBuilder& line) = 0;

  const XEventMetadata* event_metadata() const { return event_metadata_; }

 protected:
  explicit XplaneEventMutator(XEventMetadata* event_metadata)
      : event_metadata_(event_metadata) {}

  XEventMetadata* event_metadata_;
};

class XplaneEventMutatorFactory {
 public:
  virtual ~XplaneEventMutatorFactory() = default;

  virtual std::vector<std::unique_ptr<XplaneEventMutator>> CreateMutators(
      XPlaneBuilder& xplane) const = 0;

 protected:
  XplaneEventMutatorFactory() = default;
};

}  // namespace profiler
}  // namespace tsl

#endif  // MACHINA_XLATSL_PROFILER_UTILS_XPLANE_MUTATORS_H_
