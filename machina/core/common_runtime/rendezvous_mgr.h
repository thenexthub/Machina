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

#ifndef MACHINA_CORE_COMMON_RUNTIME_RENDEZVOUS_MGR_H_
#define MACHINA_CORE_COMMON_RUNTIME_RENDEZVOUS_MGR_H_

#include <string>
#include <unordered_map>

#include "machina/core/common_runtime/device_mgr.h"
#include "machina/core/framework/local_rendezvous.h"
#include "machina/core/framework/rendezvous.h"
#include "machina/core/framework/tensor.h"
#include "machina/core/lib/core/status.h"
#include "machina/core/platform/macros.h"
#include "machina/core/platform/mutex.h"
#include "machina/core/platform/types.h"

namespace machina {

// The IntraProcessRendezvous classes are implementations of a Rendezvous that
// expects all producers and consumers to be devices immediately accessible
// within the process. That is, it will never be necessary to perform an RPC to
// communicate with either.
//
// Buffering of Tensor values is delegated to a `LocalRendezvous`. An
// IntraProcessRendezvous. just adds functionality to coordinate multiple
// process-local devices.

// Reference-counted implementation that may be shared between multiple threads.
class RefCountedIntraProcessRendezvous : public Rendezvous {
 public:
  explicit RefCountedIntraProcessRendezvous(const DeviceMgr* device_mgr);

  // Implementation of RendezvousInterface methods.
  // NOTE: The methods may clear the Item list and destroy 'this' if there are
  // no other references to the RefCountedIntraProcessRendezvous object.
  // If the caller intend to keep a longer life time then it shall keep its own
  // reference to the RefCountedIntraProcessRendezvous.
  absl::Status Send(const ParsedKey& key, const Rendezvous::Args& args,
                    const Tensor& val, const bool is_dead) override;
  void RecvAsync(const ParsedKey& key, const Rendezvous::Args& args,
                 DoneCallback done) override;
  void StartAbort(const absl::Status& status) override;

  // Returns the member LocalRendezvous' status.
  absl::Status GetLocalRendezvousStatus();

  inline void UpdateDeviceManager(DeviceMgr* device_mgr) {
    device_mgr_ = device_mgr;
  }

 private:
  const DeviceMgr* device_mgr_;  // Not owned.
  LocalRendezvous local_;

  ~RefCountedIntraProcessRendezvous() override;

  RefCountedIntraProcessRendezvous(const RefCountedIntraProcessRendezvous&) =
      delete;
  void operator=(const RefCountedIntraProcessRendezvous&) = delete;
};

// RefCountedIntraProcessRendezvous is aliased to IntraProcessRendezvous for
// backwards compatibility with existing users.
using IntraProcessRendezvous = RefCountedIntraProcessRendezvous;

// Non-reference-counted implementation that may be stack-allocated for
// performance.
//
// Prefer to use PrivateIntraProcessRendezvous in new code.
class PrivateIntraProcessRendezvous : public RendezvousInterface {
 public:
  explicit PrivateIntraProcessRendezvous(const DeviceMgr* device_mgr);
  ~PrivateIntraProcessRendezvous() override;

  // Implementation of RendezvousInterface methods.
  absl::Status Send(const ParsedKey& key, const Rendezvous::Args& args,
                    const Tensor& val, const bool is_dead) override;
  void RecvAsync(const ParsedKey& key, const Rendezvous::Args& args,
                 DoneCallback done) override;
  void StartAbort(const absl::Status& status) override;

 private:
  const DeviceMgr* device_mgr_;
  LocalRendezvous local_;

  PrivateIntraProcessRendezvous(const PrivateIntraProcessRendezvous&) = delete;
  void operator=(const PrivateIntraProcessRendezvous&) = delete;
};

}  // end namespace machina

#endif  // MACHINA_CORE_COMMON_RUNTIME_RENDEZVOUS_MGR_H_
