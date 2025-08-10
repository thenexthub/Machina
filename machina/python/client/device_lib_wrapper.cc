/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, May 15, 2025.
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

#include <memory>
#include <string>
#include <vector>

#include "pybind11/pybind11.h"  // from @pybind11
#include "machina/core/common_runtime/device.h"
#include "machina/core/common_runtime/device_factory.h"
#include "machina/core/framework/device_attributes.pb.h"
#include "machina/core/lib/core/errors.h"
#include "machina/core/protobuf/config.pb.h"
#include "machina/core/public/session_options.h"
#include "machina/python/lib/core/pybind11_proto.h"
#include "machina/python/lib/core/pybind11_status.h"

namespace py = ::pybind11;

PYBIND11_MODULE(_pywrap_device_lib, m) {
  m.def("list_devices", [](py::object serialized_config) {
    machina::ConfigProto config;
    if (!serialized_config.is_none()) {
      config.ParseFromString(
          static_cast<std::string>(serialized_config.cast<py::bytes>()));
    }

    machina::SessionOptions options;
    options.config = config;
    std::vector<std::unique_ptr<machina::Device>> devices;
    machina::MaybeRaiseFromStatus(machina::DeviceFactory::AddDevices(
        options, /*name_prefix=*/"", &devices));

    py::list results;
    std::string serialized_attr;
    for (const auto& device : devices) {
      if (!device->attributes().SerializeToString(&serialized_attr)) {
        machina::MaybeRaiseFromStatus(machina::errors::Internal(
            "Could not serialize DeviceAttributes to bytes"));
      }

      // The default type caster for std::string assumes its contents
      // is UTF8-encoded.
      results.append(py::bytes(serialized_attr));
    }
    return results;
  });
}
