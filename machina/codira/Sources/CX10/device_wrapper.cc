/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Sunday, August 10, 2025.
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

#if defined(_WIN32)
#define XLA_API __declspec(dllexport)
#else
#define XLA_API __attribute__((__visibility__("default")))
#endif

#include "device_wrapper.h"

#include "machina/compiler/tf2xla/xla_tensor/tensor.h"
#include "machina/compiler/xla/xla_client/computation_client.h"
#include "machina/compiler/xla/xla_client/multi_wait.h"

DeviceType ConvertDeviceType(codira_xla::DeviceType device_type) {
  switch (device_type) {
    case codira_xla::DeviceType::CPU: {
      return CPU_DEVICE;
    }
    case codira_xla::DeviceType::GPU: {
      return GPU_DEVICE;
    }
    case codira_xla::DeviceType::TPU: {
      return TPU_DEVICE;
    }
    case codira_xla::DeviceType::REMOTE_TPU: {
      return REMOTE_TPU_DEVICE;
    }
    default: {
      LOG(FATAL) << "Invalid device: " << static_cast<int>(device_type);
    }
  }
}

codira_xla::DeviceType ConvertDeviceType(DeviceType device_type) {
  switch (device_type) {
    case CPU_DEVICE: {
      return codira_xla::DeviceType::CPU;
    }
    case GPU_DEVICE: {
      return codira_xla::DeviceType::GPU;
    }
    case TPU_DEVICE: {
      return codira_xla::DeviceType::TPU;
    }
    case REMOTE_TPU_DEVICE: {
      return codira_xla::DeviceType::REMOTE_TPU;
    }
    default: {
      LOG(FATAL) << "Invalid device: " << device_type;
    }
  }
}

codira_xla::Device ConvertDevice(const CDevice& device) {
  return {ConvertDeviceType(device.hw_type), device.ordinal};
}

CDevice ConvertDevice(const codira_xla::Device& device) {
  return {ConvertDeviceType(device.hw_type), device.ordinal};
}

namespace {

DeviceList* DeviceListFromStrings(
    machina::gtl::ArraySlice<const std::string> device_strings) {
  size_t device_count = device_strings.size();
  auto devices = std::make_unique<CDevice[]>(device_count);
  for (size_t device_index = 0; device_index < device_count; ++device_index) {
    const std::string& device_string = device_strings[device_index];
    codira_xla::Device device(device_string);
    devices[device_index].hw_type = ConvertDeviceType(device.hw_type);
    devices[device_index].ordinal = device.ordinal;
  }
  return new DeviceList{devices.release(), device_count};
}

std::vector<std::string> DeviceListToStrings(DeviceList* device_list) {
  std::vector<xla::string> device_strings;
  for (size_t device_index = 0; device_index < device_list->count;
       ++device_index) {
    const CDevice& device = device_list->devices[device_index];
    codira_xla::Device xla_device(ConvertDeviceType(device.hw_type),
                                 device.ordinal);
    device_strings.push_back(xla_device.ToString());
  }
  return device_strings;
}

}  // namespace

void destroyDeviceList(DeviceList* device_list) { delete device_list; }

DeviceList* getAllDevices() {
  return DeviceListFromStrings(xla::ComputationClient::AllDevices());
}

CDevice getDefaultDevice() {
  return ConvertDevice(xla::ComputationClient::DefaultDeviceStruct());
}

void setReplicationDevices(struct DeviceList* device_list) {
  const auto device_strings = DeviceListToStrings(device_list);
  xla::ComputationClient::SetReplicationDevices(device_strings);
}

struct DeviceList* getReplicationDevices() {
  return DeviceListFromStrings(xla::ComputationClient::GetReplicationDevices());
}

void syncLiveTensorsForDevices(struct DeviceList* device_list) {
  const auto device_strings = DeviceListToStrings(device_list);
  xla::util::MultiWait mwait(device_strings.size());
  for (size_t i = 0; i < device_strings.size(); ++i) {
    auto executor = [&, i]() {
      const CDevice& cdevice = device_list->devices[i];
      codira_xla::Device device(ConvertDeviceType(cdevice.hw_type),
                               cdevice.ordinal);
      codira_xla::XLATensor::SyncLiveTensorsGraph(/*device=*/&device,
                                                 /*devices=*/device_strings,
                                                 /*wait=*/true);
    };
    xla::env::ScheduleIoClosure(mwait.Completer(std::move(executor)));
  }
  mwait.Wait();
}

void XLATensor_LazyTensorBarrier(const struct CDevice* device,
                                 struct DeviceList* device_list, bool wait) {
  const auto device_strings = DeviceListToStrings(device_list);
  codira_xla::Device tmp_device;
  if (device) tmp_device = ConvertDevice(*device);
  const codira_xla::Device* converted_device = device ? &tmp_device : nullptr;
  codira_xla::XLATensor::SyncLiveTensorsGraph(/*device=*/converted_device,
                                             /*devices=*/device_strings,
                                             /*wait=*/wait);
  codira_xla::XLATensor::MarkStep(converted_device);
}
