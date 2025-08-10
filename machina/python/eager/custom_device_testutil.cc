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

#include "machina/c/eager/custom_device_testutil.h"

#include "Python.h"
#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/stl.h"  // from @pybind11
#include "machina/c/c_api.h"
#include "machina/c/eager/c_api.h"
#include "machina/c/eager/c_api_experimental.h"
#include "machina/c/safe_ptr.h"
#include "machina/c/tf_status.h"
#include "machina/c/tf_status_helper.h"
#include "machina/python/lib/core/py_exception_registry.h"
#include "machina/python/lib/core/pybind11_lib.h"
#include "machina/python/lib/core/pybind11_status.h"
#include "machina/python/lib/core/safe_pyobject_ptr.h"
#include "machina/python/util/util.h"

namespace py = pybind11;

void CallDelete_Flag(PyObject* capsule) {
  delete reinterpret_cast<bool*>(PyCapsule_GetPointer(capsule, "flag"));
}

void CallDelete_Device(PyObject* capsule) {
  delete reinterpret_cast<TFE_CustomDevice*>(
      PyCapsule_GetPointer(capsule, "TFE_CustomDevice"));
}

void CallDelete_DeviceInfo(PyObject* capsule) {
  PyErr_SetString(PyExc_AssertionError,
                  "Capsule should be consumed by TFE_Py_RegisterCustomDevice");
}

PYBIND11_MODULE(custom_device_testutil, m) {
  m.def("GetLoggingDeviceCapsules", [](const char* name) {
    bool* arrived_flag = new bool;
    bool* executed_flag = new bool;
    *arrived_flag = false;
    *executed_flag = false;
    machina::Safe_PyObjectPtr arrived_capsule(
        PyCapsule_New(arrived_flag, "flag", &CallDelete_Flag));
    machina::Safe_PyObjectPtr executed_capsule(
        PyCapsule_New(executed_flag, "flag", &CallDelete_Flag));
    TFE_CustomDevice* device;
    void* device_info;
    AllocateLoggingDevice(name, arrived_flag, executed_flag, &device,
                          &device_info);
    machina::Safe_PyObjectPtr device_capsule(
        PyCapsule_New(device, "TFE_CustomDevice", &CallDelete_Device));
    machina::Safe_PyObjectPtr device_info_capsule(PyCapsule_New(
        device_info, "TFE_CustomDevice_DeviceInfo", &CallDelete_DeviceInfo));
    return machina::PyoOrThrow(
        PyTuple_Pack(4, device_capsule.get(), device_info_capsule.get(),
                     arrived_capsule.get(), executed_capsule.get()));
  });
  m.def("FlagValue", [](py::capsule flag_capsule) {
    bool* flag = reinterpret_cast<bool*>(
        PyCapsule_GetPointer(flag_capsule.ptr(), "flag"));
    if (PyErr_Occurred()) throw py::error_already_set();
    return *flag;
  });
}
