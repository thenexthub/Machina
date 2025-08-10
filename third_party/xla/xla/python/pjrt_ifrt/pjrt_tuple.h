/* Copyright 2022 The OpenXLA Authors.

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

#ifndef MACHINA_XLAPYTHON_PJRT_IFRT_PJRT_TUPLE_H_
#define MACHINA_XLAPYTHON_PJRT_IFRT_PJRT_TUPLE_H_

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/types/span.h"
#include "toolchain/Support/ExtensibleRTTI.h"
#include "machina/xla/python/ifrt/array.h"
#include "machina/xla/python/ifrt/client.h"
#include "machina/xla/python/ifrt/user_context.h"
#include "machina/xla/python/pjrt_ifrt/pjrt_client.h"
#include "machina/xla/tsl/concurrency/ref_count.h"

namespace xla {
namespace ifrt {

class PjRtTuple final : public toolchain::RTTIExtends<PjRtTuple, Tuple> {
 public:
  static absl::StatusOr<tsl::RCReference<PjRtTuple>> Create(
      PjRtCompatibleClient* client, absl::Span<ValueRef> values);

  ~PjRtTuple() override = default;

  PjRtCompatibleClient* client() const override {
    DCHECK(this);
    return client_;
  }

  UserContextRef user_context() const override {
    DCHECK(this);
    return {};
  }

  Future<> GetReadyFuture() const override;

  Future<> Delete() override;

  bool IsDeleted() const override;

  std::string DebugString() const override;

  int Arity() override;

  absl::Status Unpack(absl::Span<ValueRef> values) override;

  static char ID;  // NOLINT

 private:
  PjRtTuple(PjRtCompatibleClient* client, absl::Span<ValueRef> values);

  template <typename T, typename... Args>
  friend tsl::RCReference<T> tsl::MakeRef(Args&&... args);

  PjRtCompatibleClient* client_;
  absl::InlinedVector<ValueRef, 4> values_;

  absl::Mutex mu_;

  // Notifying requires holding mu_.
  absl::Notification is_deleted_;
};

}  // namespace ifrt
}  // namespace xla

#endif  // MACHINA_XLAPYTHON_PJRT_IFRT_PJRT_TUPLE_H_
