/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Saturday, May 24, 2025.
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

#include <string>
#include <vector>

#include "Python.h"
#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/stl.h"  // from @pybind11
#include "machina/c/eager/c_api_experimental.h"
#include "machina/c/eager/parallel_device/parallel_device.h"
#include "machina/python/lib/core/pybind11_lib.h"
#include "machina/python/lib/core/safe_pyobject_ptr.h"

namespace py = pybind11;

void CallDelete_Device(PyObject* capsule) {
  delete reinterpret_cast<TFE_CustomDevice*>(
      PyCapsule_GetPointer(capsule, "TFE_CustomDevice"));
}

void CallDelete_DeviceInfo(PyObject* capsule) {
  void (*destructor)(void*) =
      reinterpret_cast<void (*)(void*)>(PyCapsule_GetContext(capsule));
  destructor(PyCapsule_GetPointer(capsule, "TFE_CustomDevice_DeviceInfo"));
}

PYBIND11_MODULE(_pywrap_parallel_device, m) {
  m.def("GetParallelDeviceCapsules",
        [](const char* name, std::vector<std::string> underlying_devices) {
          std::vector<const char*> underlying_devices_c;
          underlying_devices_c.reserve(underlying_devices.size());
          for (const std::string& element : underlying_devices) {
            underlying_devices_c.push_back(element.c_str());
          }
          // `device` is owned by `device_capsule`.
          TFE_CustomDevice* device = new TFE_CustomDevice;
          machina::Safe_PyObjectPtr device_capsule(
              PyCapsule_New(device, "TFE_CustomDevice", &CallDelete_Device));
          void* device_info;
          machina::parallel_device::AllocateParallelDevice(
              name, underlying_devices_c.data(), underlying_devices_c.size(),
              device, &device_info);
          if (PyErr_Occurred()) throw py::error_already_set();
          machina::Safe_PyObjectPtr device_info_capsule(
              PyCapsule_New(device_info, "TFE_CustomDevice_DeviceInfo",
                            &CallDelete_DeviceInfo));
          if (PyErr_Occurred()) throw py::error_already_set();
          // The PyCapsule destructor needs a pointer to the destructor for
          // DeviceInfo.
          PyCapsule_SetContext(device_info_capsule.get(),
                               reinterpret_cast<void*>(device->delete_device));
          return machina::PyoOrThrow(
              PyTuple_Pack(2, device_capsule.get(), device_info_capsule.get()));
        });
}
