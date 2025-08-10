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

#ifndef MACHINA_MACHINA_XLA_SERVICE_GPU_TRANSFORMS_STREAM_ATTRIBUTE_ANNOTATOR_H_
#define MACHINA_MACHINA_XLA_SERVICE_GPU_TRANSFORMS_STREAM_ATTRIBUTE_ANNOTATOR_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "machina/xla/hlo/ir/hlo_computation.h"
#include "machina/xla/hlo/ir/hlo_module.h"
#include "machina/xla/hlo/pass/hlo_pass_interface.h"
#include "machina/xla/stream_executor/device_description.h"

namespace xla::gpu {

// This pass checks to see if:
//  1.  there's any instruction, that
//      consumes data from other computes streams,
//      is missing "wait_on_operation_queues" attribute.
//  2.  there's any fusion instruction with non-default
//      stream fusion root.
// It will annotate the corresponding instruction with
// the correct attribute in GpuBackendConfig.
// Instructions annotated with operation_queue_id > 0
// will be wrapped with AsyncInstruction and split into
// AsyncStart and AsyncDone in the
// StreamAttributeAsyncWrapper pass.
// We also check if there's any non-default-stream
// instruction's user doesn't have the correct "wait_on_operation_queues"
// attribute and set it with producer's operation_queue_id.
// "wait_on_operation_queues" will need to used by the emitter to emit the
// correct WaitForStreams thunk.

class StreamAttributeAnnotator : public HloModulePass {
 public:
  explicit StreamAttributeAnnotator(
      const se::DeviceDescription& device_description)
      : device_description_(device_description) {}

  absl::string_view name() const override {
    return "stream-attribute-annotator";
  }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  const se::DeviceDescription& device_description_;
};

}  // namespace xla::gpu

#endif  // MACHINA_MACHINA_XLA_SERVICE_GPU_TRANSFORMS_STREAM_ATTRIBUTE_ANNOTATOR_H_
