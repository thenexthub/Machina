/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 26, 2025.
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

#include "machina/core/distributed_runtime/remote_device.h"

#include <stdlib.h>

#include <vector>

#include "machina/core/common_runtime/device.h"
#include "machina/core/common_runtime/process_util.h"
#include "machina/core/common_runtime/renamed_device.h"
#include "machina/core/distributed_runtime/worker_cache.h"
#include "machina/core/distributed_runtime/worker_interface.h"
#include "machina/core/lib/core/errors.h"
#include "machina/core/lib/gtl/cleanup.h"
#include "machina/core/platform/logging.h"
#include "machina/core/platform/macros.h"
#include "machina/core/protobuf/worker.pb.h"
#include "machina/core/util/device_name_utils.h"

namespace machina {

class RemoteDevice : public Device {
 public:
  RemoteDevice(Env* env, const DeviceAttributes& da)
      : Device(env, da),
        local_dev_name_(DeviceNameUtils::LocalName(da.name())) {}

  absl::Status Sync() override { return absl::OkStatus(); }
  Allocator* GetAllocator(AllocatorAttributes attr) override { return nullptr; }

  ResourceMgr* resource_manager() override {
    LOG(FATAL) << "Accessing the resource manager of a remote device is not "
               << "supported.";
    std::abort();
  }

  bool IsLocal() const override { return false; }

  bool IsRemoteCallAllowed() const override { return true; }

 private:
  const string local_dev_name_;

  RemoteDevice(const RemoteDevice&) = delete;
  void operator=(const RemoteDevice&) = delete;
};

void AsRemoteDevices(
    Env* env,
    const protobuf::RepeatedPtrField<DeviceAttributes>& device_attributes,
    LookupLocalDevice lookup_local_device,
    std::vector<std::unique_ptr<Device>>* remote_devices) {
  for (const auto& da : device_attributes) {
    Device* local_device;
    if (lookup_local_device != nullptr &&
        lookup_local_device(da.name(), &local_device).ok()) {
      remote_devices->emplace_back(RenamedDevice::NewRenamedDevice(
          local_device->name(), local_device, false, false));
    } else {
      auto d = new RemoteDevice(env, da);
      remote_devices->emplace_back(d);
    }
  }
}

void NewRemoteDevices(Env* env, WorkerCacheInterface* worker_cache,
                      const string& worker_name, NewRemoteDevicesDone done) {
  WorkerInterface* wi = worker_cache->GetOrCreateWorker(worker_name);
  if (wi == nullptr) {
    std::vector<Device*> empty;
    done(errors::NotFound("Device ", worker_name, " is not found."), &empty);
    return;
  }
  struct Call {
    GetStatusRequest req;
    GetStatusResponse resp;
  };
  Call* call = new Call;
  auto cb = [env, worker_cache, worker_name, done, wi,
             call](const absl::Status& status) {
    absl::Status s = status;
    std::vector<Device*> remote_devices;
    auto cleanup = gtl::MakeCleanup(
        [&worker_cache, &worker_name, &wi, &done, &remote_devices, &s, call] {
          worker_cache->ReleaseWorker(worker_name, wi);
          done(s, &remote_devices);
          delete call;
        });
    if (!s.ok()) {
      return;
    }
    DeviceNameUtils::ParsedName worker_name_parsed;
    if (!DeviceNameUtils::ParseFullName(worker_name, &worker_name_parsed) ||
        !worker_name_parsed.has_job || !worker_name_parsed.has_replica ||
        !worker_name_parsed.has_task) {
      s = errors::InvalidArgument("Could not parse worker name: ", worker_name);
      LOG(WARNING) << s;
      return;
    }
    remote_devices.reserve(call->resp.device_attributes_size());
    for (const DeviceAttributes& da : call->resp.device_attributes()) {
      DeviceNameUtils::ParsedName device_name_parsed;
      CHECK(DeviceNameUtils::ParseFullName(da.name(), &device_name_parsed))
          << "Device attribute name '" << da.name() << "' could not be "
          << "parsed. Device Attribute: " << da.DebugString();
      // Preserve the exact name, if possible.
      // TODO(b/37868888): Simplify when legacy device name formats removed.
      if (device_name_parsed.job == worker_name_parsed.job &&
          device_name_parsed.replica == worker_name_parsed.replica &&
          device_name_parsed.task == worker_name_parsed.task) {
        auto d = new RemoteDevice(env, da);
        remote_devices.push_back(d);
      } else {
        DeviceAttributes da_rewritten = da;
        da_rewritten.set_name(DeviceNameUtils::FullName(
            worker_name_parsed.job, worker_name_parsed.replica,
            worker_name_parsed.task, device_name_parsed.type,
            device_name_parsed.id));
        auto d = new RemoteDevice(env, da_rewritten);

        // Experimental: Skipping over adding any TPU-type devices that aren't
        // on the job called "worker" (but still adds the CPUs of other jobs).
        if (getenv("TPU_NO_POPULATE_DEVICE_LIST_FROM_CLUSTER_SPEC") !=
            nullptr) {
          if (worker_name_parsed.job == "worker" ||
              device_name_parsed.type.find("TPU") == std::string::npos) {
            remote_devices.push_back(d);
          }
        } else {
          remote_devices.push_back(d);
        }
      }
    }
  };
  wi->GetStatusAsync(/*opts=*/nullptr, &call->req, &call->resp,
                     /*fail_fast=*/false, cb);
}

std::unique_ptr<Device> NewRemoteDevice(Env* env,
                                        DeviceAttributes device_attribute) {
  return std::make_unique<RemoteDevice>(env, device_attribute);
}

}  // namespace machina
