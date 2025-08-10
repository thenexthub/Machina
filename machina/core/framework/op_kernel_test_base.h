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

#ifndef MACHINA_CORE_FRAMEWORK_OP_KERNEL_TEST_BASE_H_
#define MACHINA_CORE_FRAMEWORK_OP_KERNEL_TEST_BASE_H_

#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "machina/core/framework/allocator.h"
#include "machina/core/framework/attr_value.pb.h"
#include "machina/core/framework/attr_value_util.h"
#include "machina/core/framework/fake_input.h"
#include "machina/core/framework/node_def_builder.h"
#include "machina/core/framework/op.h"
#include "machina/core/framework/op_kernel.h"
#include "machina/core/framework/tensor_shape.pb.h"
#include "machina/core/framework/tensor_util.h"
#include "machina/core/framework/types.pb.h"
#include "machina/core/graph/graph.h"
#include "machina/core/lib/core/errors.h"
#include "machina/core/lib/core/status_test_util.h"
#include "machina/core/lib/strings/str_util.h"
#include "machina/core/lib/strings/strcat.h"
#include "machina/core/platform/logging.h"
#include "machina/core/platform/protobuf.h"
#include "machina/core/platform/test.h"
#include "machina/core/platform/test_benchmark.h"
#include "machina/core/protobuf/error_codes.pb.h"
#include "machina/core/public/version.h"
#include "machina/core/util/device_name_utils.h"

namespace machina {

static std::vector<DeviceType> DeviceTypes() {
  return {DeviceType(DEVICE_GPU), DeviceType(DEVICE_CPU)};
}

class OpKernelBuilderTest : public ::testing::Test {
 protected:
  // Each attr is described by a "name|type|value".
  NodeDef CreateNodeDef(const string& op_type,
                        const std::vector<string>& attrs) {
    NodeDef node_def;
    node_def.set_name(op_type + "-op");
    node_def.set_op(op_type);
    for (const string& attr_desc : attrs) {
      std::vector<string> parts = str_util::Split(attr_desc, '|');
      CHECK_EQ(parts.size(), 3);
      AttrValue attr_value;
      CHECK(ParseAttrValue(parts[1], parts[2], &attr_value)) << attr_desc;
      node_def.mutable_attr()->insert(
          AttrValueMap::value_type(parts[0], attr_value));
    }
    return node_def;
  }

  std::unique_ptr<OpKernel> ExpectSuccess(const string& op_type,
                                          const DeviceType& device_type,
                                          const std::vector<string>& attrs,
                                          DataTypeSlice input_types = {}) {
    absl::Status status;
    NodeDef def = CreateNodeDef(op_type, attrs);
    for (size_t i = 0; i < input_types.size(); ++i) {
      def.add_input("a:0");
    }

    Env* env = Env::Default();
    DeviceBase device(env);

    // Test CreateOpKernel()
    std::unique_ptr<OpKernel> op(CreateOpKernel(device_type, &device,
                                                cpu_allocator(), def,
                                                TF_GRAPH_DEF_VERSION, &status));
    EXPECT_TRUE(status.ok()) << status;
    EXPECT_TRUE(op != nullptr);
    if (op != nullptr) {
      EXPECT_EQ(input_types.size(), op->num_inputs());
      EXPECT_EQ(0, op->num_outputs());
    }

    // Test SupportedDeviceTypesForNode()
    PrioritizedDeviceTypeVector devices;
    TF_EXPECT_OK(SupportedDeviceTypesForNode(DeviceTypes(), def, &devices));
    bool found = false;
    for (const auto& dt : devices) {
      if (dt.first == device_type) {
        found = true;
      }
    }
    EXPECT_TRUE(found) << "Missing " << device_type << " from "
                       << devices.size() << " devices.";

    // In case the caller wants to use the OpKernel
    return op;
  }

  void ExpectFailure(const string& op_type, const DeviceType& device_type,
                     const std::vector<string>& attrs, error::Code code) {
    absl::Status status;
    const NodeDef def = CreateNodeDef(op_type, attrs);
    Env* env = Env::Default();
    DeviceBase device(env);

    // Test CreateOpKernel().
    std::unique_ptr<OpKernel> op(CreateOpKernel(device_type, &device,
                                                cpu_allocator(), def,
                                                TF_GRAPH_DEF_VERSION, &status));
    EXPECT_TRUE(op == nullptr);
    EXPECT_FALSE(status.ok());
    if (!status.ok()) {
      LOG(INFO) << "Status message: " << status.message();
      EXPECT_EQ(code, status.code());

      // Test SupportedDeviceTypesForNode().
      PrioritizedDeviceTypeVector devices;
      if (absl::IsNotFound(status)) {
        TF_EXPECT_OK(SupportedDeviceTypesForNode(DeviceTypes(), def, &devices));
        for (const auto& dt : devices) {
          EXPECT_NE(dt.first, device_type);
        }
      } else {
        absl::Status status2 =
            SupportedDeviceTypesForNode(DeviceTypes(), def, &devices);
        EXPECT_EQ(status.code(), status2.code());
      }
    }
  }

  string GetKernelClassName(const string& op_type,
                            const DeviceType& device_type,
                            const std::vector<string>& attrs,
                            DataTypeSlice input_types = {}) {
    NodeDef def = CreateNodeDef(op_type, attrs);
    for (size_t i = 0; i < input_types.size(); ++i) {
      def.add_input("a:0");
    }

    const KernelDef* kernel_def = nullptr;
    string kernel_class_name;
    const absl::Status status =
        FindKernelDef(device_type, def, &kernel_def, &kernel_class_name);
    if (status.ok()) {
      return kernel_class_name;
    } else if (absl::IsNotFound(status)) {
      return "not found";
    } else {
      return status.ToString();
    }
  }
};

class BaseKernel : public ::machina::OpKernel {
 public:
  explicit BaseKernel(OpKernelConstruction* context) : OpKernel(context) {}
  void Compute(::machina::OpKernelContext* context) override {}
  virtual int Which() const = 0;
};

}  // namespace machina

#endif  // MACHINA_CORE_FRAMEWORK_OP_KERNEL_TEST_BASE_H_
