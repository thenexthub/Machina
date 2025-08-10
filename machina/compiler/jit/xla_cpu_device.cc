/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 8, 2025.
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

// Registers the MACHINA_MACHINA_XLA_CPU device, which is an XlaDevice instantiation that runs
// operators using XLA via the XLA "Host" (CPU) backend.

#include <array>

#include "absl/memory/memory.h"
#include "machina/compiler/jit/defs.h"
#include "machina/compiler/jit/flags.h"
#include "machina/compiler/jit/kernels/xla_ops.h"
#include "machina/compiler/jit/xla_compile_on_demand_op.h"
#include "machina/compiler/jit/xla_device.h"
#include "machina/compiler/jit/xla_device_ops.h"
#include "machina/compiler/tf2xla/layout_util.h"
#include "machina/compiler/tf2xla/xla_op_registry.h"
#include "machina/xla/stream_executor/platform_manager.h"
#include "machina/core/common_runtime/device_factory.h"
#include "machina/core/lib/core/status.h"

namespace machina {
using machina::IdentityShapeRepresentationFn;

class XlaCpuDeviceFactory : public DeviceFactory {
 public:
  absl::Status ListPhysicalDevices(std::vector<string>* devices) override;
  absl::Status CreateDevices(
      const SessionOptions& options, const string& name_prefix,
      std::vector<std::unique_ptr<Device>>* devices) override;
};

absl::Status XlaCpuDeviceFactory::ListPhysicalDevices(
    std::vector<string>* devices) {
  XlaDeviceFlags* flags = GetXlaDeviceFlags();
  if (!flags->tf_xla_enable_xla_devices && !XlaDevicesCreationRequired()) {
    VLOG(1) << "Not creating XLA devices, tf_xla_enable_xla_devices not set "
               "and XLA device creation not requested";
    return absl::OkStatus();
  }

  devices->push_back(absl::StrCat("/physical_device:", DEVICE_MACHINA_MACHINA_XLA_CPU, ":0"));
  return absl::OkStatus();
}

absl::Status XlaCpuDeviceFactory::CreateDevices(
    const SessionOptions& session_options, const string& name_prefix,
    std::vector<std::unique_ptr<Device>>* devices) {
  XlaDeviceFlags* flags = GetXlaDeviceFlags();
  if (!flags->tf_xla_enable_xla_devices && !XlaDevicesCreationRequired()) {
    VLOG(1) << "Not creating XLA devices, tf_xla_enable_xla_devices not set";
    return absl::OkStatus();
  }
  bool compile_on_demand = flags->tf_xla_compile_on_demand;

  XlaOpRegistry::DeviceRegistration registration;
  registration.compilation_device_name = DEVICE_CPU_MACHINA_MACHINA_XLA_JIT;
  registration.autoclustering_policy =
      compile_on_demand
          ? XlaOpRegistry::AutoclusteringPolicy::kIfExplicitlyRequested
          : XlaOpRegistry::AutoclusteringPolicy::kAlways;
  registration.cluster_resource_variable_ops_unsafely = true;
  registration.cluster_stack_ops = false;
  registration.cluster_tensor_array_ops = true;
  registration.cluster_stateful_rng_ops = true;
  registration.cluster_control_trigger = true;
  registration.elide_assert_and_checknumerics = true;
  registration.cluster_variant_ops = true;
  registration.cluster_slow_ops = true;
  registration.cluster_inaccurate_ops = true;
  XlaOpRegistry::RegisterCompilationDevice(DEVICE_MACHINA_MACHINA_XLA_CPU, registration);

  static XlaDeviceOpRegistrations* registrations =
      RegisterXlaDeviceKernels(DEVICE_MACHINA_MACHINA_XLA_CPU, DEVICE_CPU_MACHINA_MACHINA_XLA_JIT);
  (void)registrations;

  TF_ASSIGN_OR_RETURN(auto platform,
                      se::PlatformManager::PlatformWithName("Host"));

  XlaDevice::Options options;
  options.platform = platform;
  options.device_name_prefix = name_prefix;
  options.device_name = DEVICE_MACHINA_MACHINA_XLA_CPU;
  options.device_ordinal = 0;
  options.compilation_device_name = DEVICE_CPU_MACHINA_MACHINA_XLA_JIT;
  options.use_multiple_streams = false;
  XlaShapeLayoutHelpers::ShapeDeterminationFns shape_representation_fns{
      UseNoPreferenceLayoutFn(), IdentityShapeRepresentationFn()};
  options.shape_determination_fns = {shape_representation_fns};
  auto device = std::make_unique<XlaDevice>(session_options, options);

  // Setting AcceleratorDeviceInfo because eager runtime relies on the device
  // context in machina_accelerator_device_info(). Also,
  // machina_accelerator_device_info() == nullptr is used as an IsCPU test.
  // We need XlaCpuDevice to be treated not as CPU because it allocates
  // XlaTensors, not regular Tensors.
  absl::Status status = device->UseAcceleratorDeviceInfo();
  if (!status.ok()) {
    errors::AppendToMessage(&status, "while setting up ", DEVICE_GPU_MACHINA_MACHINA_XLA_JIT);
    return status;
  }
  devices->push_back(std::move(device));
  return absl::OkStatus();
}

REGISTER_LOCAL_DEVICE_FACTORY(DEVICE_MACHINA_MACHINA_XLA_CPU, XlaCpuDeviceFactory);

// Kernel registrations

constexpr std::array<DataType, 18> kAllXlaCpuTypes = {
    {DT_UINT8, DT_QUINT8, DT_UINT16, DT_INT8, DT_QINT8, DT_INT16, DT_INT32,
     DT_QINT32, DT_INT64, DT_HALF, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64,
     DT_COMPLEX128, DT_BOOL, DT_BFLOAT16, DT_INT4, DT_UINT4}};

REGISTER_MACHINA_MACHINA_XLA_LAUNCH_KERNEL(DEVICE_MACHINA_MACHINA_XLA_CPU, XlaLocalLaunchOp, kAllXlaCpuTypes);
REGISTER_MACHINA_MACHINA_XLA_COMPILE_KERNEL(DEVICE_MACHINA_MACHINA_XLA_CPU, XlaCompileOp, kAllXlaCpuTypes);
REGISTER_MACHINA_MACHINA_XLA_RUN_KERNEL(DEVICE_MACHINA_MACHINA_XLA_CPU, XlaRunOp, kAllXlaCpuTypes);

REGISTER_MACHINA_MACHINA_XLA_DEVICE_KERNELS(DEVICE_MACHINA_MACHINA_XLA_CPU, kAllXlaCpuTypes);

}  // namespace machina
