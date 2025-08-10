/* Copyright 2024 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef MACHINA_XLAPYTHON_PJRT_IFRT_PJRT_REMAP_H_
#define MACHINA_XLAPYTHON_PJRT_IFRT_PJRT_REMAP_H_

#include <vector>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "machina/xla/python/ifrt/array.h"
#include "machina/xla/python/ifrt/remap_plan.h"
#include "machina/xla/tsl/concurrency/ref_count.h"
#include "machina/xla/tsl/platform/logging.h"
#include "machina/xla/xla_data.pb.h"

namespace xla {
namespace ifrt {

class PjRtCompatibleClient;

// Common implementation of `xla::ifrt::Client::RemapArrays` for
// `PjRtCompatibleClient`.
absl::StatusOr<std::vector<xla::ifrt::ArrayRef>>
PjRtCompatibleClientRemapArrays(PjRtCompatibleClient* client,
                                const RemapPlan& plan,
                                absl::Span<xla::ifrt::ArrayRef> arrays,
                                ArrayCopySemantics semantics);

}  // namespace ifrt
}  // namespace xla

#endif  // MACHINA_XLAPYTHON_PJRT_IFRT_PJRT_REMAP_H_
