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

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include "machina/compiler/jit/flags.h"
#include "machina/compiler/tf2xla/xla_op_registry.h"
#include "machina/xla/tsl/lib/core/status_test_util.h"
#include "machina/xla/tsl/platform/status.h"
#include "machina/core/framework/allocator.h"
#include "machina/core/framework/device_base.h"
#include "machina/core/framework/device_factory.h"
#include "machina/core/framework/fake_input.h"
#include "machina/core/framework/node_def_builder.h"
#include "machina/core/framework/tensor.h"
#include "machina/core/framework/tensor_shape.h"
#include "machina/core/framework/tensor_testutil.h"
#include "machina/core/framework/types.h"
#include "machina/core/framework/types.pb.h"
#include "machina/core/kernels/ops_testutil.h"
#include "machina/core/platform/logging.h"
#include "machina/core/platform/test.h"
#include "machina/core/util/port.h"

namespace machina {
namespace {

static bool Initialized = [] {
  machina::GetXlaDeviceFlags()->tf_xla_enable_xla_devices = true;
  return true;
}();

class UnaryOpsCompositionTest : public OpsTestBase {
 protected:
  template <typename T>
  void RunComposedOp(const std::vector<string> op_names, T input_scalar_value,
                     T expected_scalar_value) {
    string xla_device_name =
        machina::IsGoogleCudaEnabled() ? DEVICE_MACHINA_MACHINA_XLA_GPU : DEVICE_MACHINA_MACHINA_XLA_CPU;
    SetDevice(DeviceType(xla_device_name),
              std::unique_ptr<machina::Device>(DeviceFactory::NewDevice(
                  xla_device_name, {}, "/job:a/replica:0/task:0")));

    TF_ASSERT_OK(NodeDefBuilder("unary_op_composition", "_UnaryOpsComposition")
                     .Input(FakeInput(DataTypeToEnum<T>::v()))
                     .Attr("T", DataTypeToEnum<T>::v())
                     .Attr("op_names", op_names)
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());

    // We're using an XLA device here which allocates XlaTensors.  We can't
    // inspect XlaTensors directly so we create the input on the host and copy
    // it over to the XLA device.  We do the inverse on the output.

    TensorShape shape({});

    AllocatorAttributes host_alloc_attrs;
    host_alloc_attrs.set_gpu_compatible(true);
    host_alloc_attrs.set_on_host(true);
    Allocator* cpu_allocator = device_->GetAllocator(host_alloc_attrs);

    DataType dtype = DataTypeToEnum<T>::value;

    Tensor input_on_host(cpu_allocator, dtype, shape);
    test::FillValues<T>(&input_on_host, {input_scalar_value});

    Tensor* input = AddInput(dtype, shape);

    DeviceContext* device_context =
        device_->machina_accelerator_device_info()->default_context;

    TF_CHECK_OK(device_context->CopyCPUTensorToDeviceSync(&input_on_host,
                                                          device_, input));

    TF_ASSERT_OK(RunOpKernel());

    Tensor expected_tensor(cpu_allocator, dtype, shape);
    test::FillValues<T>(&expected_tensor, {expected_scalar_value});

    Tensor* output = GetOutput(0);
    Tensor output_on_host(cpu_allocator, output->dtype(), output->shape());

    TF_CHECK_OK(device_context->CopyDeviceTensorToCPUSync(
        output, "output 0", device_, &output_on_host));

    test::ExpectClose(expected_tensor, output_on_host, /*atol=*/1e-5,
                      /*rtol=*/1e-5);
  }
};

TEST_F(UnaryOpsCompositionTest, Compose_Sqrt_Sqrt_F) {
  RunComposedOp<float>({"Sqrt", "Sqrt"}, 81.0, 3.0);
}

TEST_F(UnaryOpsCompositionTest, Compose_Sqrt_Sqrt_D) {
  RunComposedOp<double>({"Sqrt", "Sqrt"}, 81.0, 3.0);
}

TEST_F(UnaryOpsCompositionTest, Compose_Sqrt_Sin_F) {
  RunComposedOp<float>({"Sqrt", "Sin"}, 81.0, std::sin(9.0f));
}

TEST_F(UnaryOpsCompositionTest, Compose_Cos_Acos_F) {
  RunComposedOp<float>({"Cos", "Acos"}, 0.5, std::acos(std::cos(0.5f)));
}

TEST_F(UnaryOpsCompositionTest, Compose_Tanh_Relu_F) {
  RunComposedOp<float>({"Tanh", "Relu"}, 0.5, std::max(0.0f, std::tanh(0.5f)));
}

TEST_F(UnaryOpsCompositionTest, Compose_Tanh_Relu_D) {
  RunComposedOp<double>({"Tanh", "Relu"}, 0.5, std::max(0.0, std::tanh(0.5)));
}

TEST_F(UnaryOpsCompositionTest, Compose_Tanh_Relu6_F) {
  RunComposedOp<float>({"Relu6"}, 11.0f, 6.0f);
}
}  // namespace
}  // end namespace machina
