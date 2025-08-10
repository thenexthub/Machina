/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 29, 2025.
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

#include "machina/c/eager/c_api_test_util.h"

#include <memory>
#include <string>
#include <vector>

#include "machina/c/eager/c_api.h"
#include "machina/c/eager/c_api_experimental.h"
#include "machina/c/eager/c_api_internal.h"
#include "machina/c/tf_datatype.h"
#include "machina/c/tf_tensor.h"
#include "machina/core/platform/logging.h"
#include "machina/core/platform/strcat.h"
#include "machina/core/platform/test.h"
#include "machina/core/platform/tstring.h"
#include "machina/core/protobuf/cluster.pb.h"
#include "machina/core/protobuf/config.pb.h"
#include "machina/core/protobuf/rewriter_config.pb.h"
#include "machina/core/util/port.h"

using machina::string;
using machina::tstring;

TFE_TensorHandle* TestScalarTensorHandle(TFE_Context* ctx, float value) {
  float data[] = {value};
  TF_Status* status = TF_NewStatus();
  TF_Tensor* t = TFE_AllocateHostTensor(ctx, TF_FLOAT, nullptr, 0, status);
  memcpy(TF_TensorData(t), &data[0], TF_TensorByteSize(t));
  TFE_TensorHandle* th = TFE_NewTensorHandleFromTensor(ctx, t, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteTensor(t);
  TF_DeleteStatus(status);
  return th;
}

TFE_TensorHandle* TestScalarTensorHandle(TFE_Context* ctx,
                                         const machina::tstring& value) {
  TF_Status* status = TF_NewStatus();
  TF_Tensor* t = TFE_AllocateHostTensor(ctx, TF_STRING, nullptr, 0, status);
  tstring* data = static_cast<tstring*>(TF_TensorData(t));
  *data = value;
  TFE_TensorHandle* th = TFE_NewTensorHandleFromTensor(ctx, t, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteTensor(t);
  TF_DeleteStatus(status);
  return th;
}

TFE_TensorHandle* TestScalarTensorHandle(TFE_Context* ctx, int value) {
  int data[] = {value};
  TF_Status* status = TF_NewStatus();
  TF_Tensor* t = TFE_AllocateHostTensor(ctx, TF_INT32, nullptr, 0, status);
  memcpy(TF_TensorData(t), &data[0], TF_TensorByteSize(t));
  TFE_TensorHandle* th = TFE_NewTensorHandleFromTensor(ctx, t, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteTensor(t);
  TF_DeleteStatus(status);
  return th;
}

TFE_TensorHandle* TestScalarTensorHandle(TFE_Context* ctx, bool value) {
  bool data[] = {value};
  TF_Status* status = TF_NewStatus();
  TF_Tensor* t = TFE_AllocateHostTensor(ctx, TF_BOOL, nullptr, 0, status);
  memcpy(TF_TensorData(t), &data[0], TF_TensorByteSize(t));
  TFE_TensorHandle* th = TFE_NewTensorHandleFromTensor(ctx, t, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteTensor(t);
  TF_DeleteStatus(status);
  return th;
}

TFE_TensorHandle* DoubleTestMatrixTensorHandle(TFE_Context* ctx) {
  int64_t dims[] = {2, 2};
  double data[] = {1.0, 2.0, 3.0, 4.0};
  TF_Status* status = TF_NewStatus();
  TF_Tensor* t = TFE_AllocateHostTensor(ctx, TF_DOUBLE, &dims[0],
                                        sizeof(dims) / sizeof(int64_t), status);
  memcpy(TF_TensorData(t), &data[0], TF_TensorByteSize(t));
  TFE_TensorHandle* th = TFE_NewTensorHandleFromTensor(ctx, t, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteTensor(t);
  TF_DeleteStatus(status);
  return th;
}

TFE_TensorHandle* TestMatrixTensorHandle(TFE_Context* ctx) {
  int64_t dims[] = {2, 2};
  float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
  TF_Status* status = TF_NewStatus();
  TF_Tensor* t = TFE_AllocateHostTensor(ctx, TF_FLOAT, &dims[0],
                                        sizeof(dims) / sizeof(int64_t), status);
  memcpy(TF_TensorData(t), &data[0], TF_TensorByteSize(t));
  TFE_TensorHandle* th = TFE_NewTensorHandleFromTensor(ctx, t, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteTensor(t);
  TF_DeleteStatus(status);
  return th;
}

TFE_TensorHandle* TestMatrixTensorHandleWithInput(TFE_Context* ctx,
                                                  float data[], int64_t dims[],
                                                  int num_dims) {
  TF_Status* status = TF_NewStatus();
  TF_Tensor* t =
      TFE_AllocateHostTensor(ctx, TF_FLOAT, &dims[0], num_dims, status);
  memcpy(TF_TensorData(t), &data[0], TF_TensorByteSize(t));
  TFE_TensorHandle* th = TFE_NewTensorHandleFromTensor(ctx, t, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteTensor(t);
  TF_DeleteStatus(status);
  return th;
}

TFE_TensorHandle* TestTensorHandleWithDimsFloat(TFE_Context* ctx, float data[],
                                                int64_t dims[], int num_dims) {
  TF_Status* status = TF_NewStatus();
  TF_Tensor* t =
      TFE_AllocateHostTensor(ctx, TF_FLOAT, &dims[0], num_dims, status);
  memcpy(TF_TensorData(t), &data[0], TF_TensorByteSize(t));
  TFE_TensorHandle* th = TFE_NewTensorHandleFromTensor(ctx, t, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteTensor(t);
  TF_DeleteStatus(status);
  return th;
}

TFE_TensorHandle* TestTensorHandleWithDimsInt(TFE_Context* ctx, int data[],
                                              int64_t dims[], int num_dims) {
  TF_Status* status = TF_NewStatus();
  TF_Tensor* t =
      TFE_AllocateHostTensor(ctx, TF_INT32, &dims[0], num_dims, status);
  memcpy(TF_TensorData(t), &data[0], TF_TensorByteSize(t));
  TFE_TensorHandle* th = TFE_NewTensorHandleFromTensor(ctx, t, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteTensor(t);
  TF_DeleteStatus(status);
  return th;
}

TFE_TensorHandle* TestMatrixTensorHandle100x100(TFE_Context* ctx) {
  constexpr int64_t dims[] = {100, 100};
  constexpr int num_elements = dims[0] * dims[1];
  float data[num_elements];
  for (int i = 0; i < num_elements; ++i) {
    data[i] = 1.0f;
  }
  TF_Status* status = TF_NewStatus();
  TF_Tensor* t = TFE_AllocateHostTensor(ctx, TF_FLOAT, &dims[0],
                                        sizeof(dims) / sizeof(int64_t), status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  memcpy(TF_TensorData(t), &data[0], TF_TensorByteSize(t));
  TFE_TensorHandle* th = TFE_NewTensorHandleFromTensor(ctx, t, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteTensor(t);
  TF_DeleteStatus(status);
  return th;
}

TFE_TensorHandle* DoubleTestMatrixTensorHandle3X2(TFE_Context* ctx) {
  int64_t dims[] = {3, 2};
  double data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  TF_Status* status = TF_NewStatus();
  TF_Tensor* t = TFE_AllocateHostTensor(ctx, TF_FLOAT, &dims[0],
                                        sizeof(dims) / sizeof(int64_t), status);
  memcpy(TF_TensorData(t), &data[0], TF_TensorByteSize(t));
  TFE_TensorHandle* th = TFE_NewTensorHandleFromTensor(ctx, t, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteTensor(t);
  TF_DeleteStatus(status);
  return th;
}

TFE_TensorHandle* TestMatrixTensorHandle3X2(TFE_Context* ctx) {
  int64_t dims[] = {3, 2};
  float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  TF_Status* status = TF_NewStatus();
  TF_Tensor* t = TFE_AllocateHostTensor(ctx, TF_FLOAT, &dims[0],
                                        sizeof(dims) / sizeof(int64_t), status);
  memcpy(TF_TensorData(t), &data[0], TF_TensorByteSize(t));
  TFE_TensorHandle* th = TFE_NewTensorHandleFromTensor(ctx, t, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteTensor(t);
  TF_DeleteStatus(status);
  return th;
}

TFE_TensorHandle* TestVariable(TFE_Context* ctx, float value,
                               const machina::string& device_name) {
  TF_Status* status = TF_NewStatus();
  // Create the variable handle.
  TFE_Op* op = TFE_NewOp(ctx, "VarHandleOp", status);
  if (TF_GetCode(status) != TF_OK) return nullptr;
  TFE_OpSetAttrType(op, "dtype", TF_FLOAT);
  TFE_OpSetAttrShape(op, "shape", {}, 0, status);
  TFE_OpSetAttrString(op, "container", "localhost", 0);
  TFE_OpSetAttrString(op, "shared_name", "", 0);
  if (!device_name.empty()) {
    TFE_OpSetDevice(op, device_name.c_str(), status);
  }
  if (TF_GetCode(status) != TF_OK) return nullptr;
  TFE_TensorHandle* var_handle = nullptr;
  int num_retvals = 1;
  TFE_Execute(op, &var_handle, &num_retvals, status);
  if (TF_GetCode(status) != TF_OK) return nullptr;
  TFE_DeleteOp(op);
  if (TF_GetCode(status) != TF_OK) return nullptr;
  CHECK_EQ(1, num_retvals);

  // Assign 'value' to it.
  op = TFE_NewOp(ctx, "AssignVariableOp", status);
  if (TF_GetCode(status) != TF_OK) return nullptr;
  TFE_OpSetAttrType(op, "dtype", TF_FLOAT);
  TFE_OpAddInput(op, var_handle, status);

  // Convert 'value' to a TF_Tensor then a TFE_TensorHandle.
  std::unique_ptr<TF_Tensor, decltype(&TF_DeleteTensor)> t(
      TF_AllocateTensor(TF_FLOAT, nullptr, 0, sizeof(value)), TF_DeleteTensor);
  memcpy(TF_TensorData(t.get()), &value, TF_TensorByteSize(t.get()));

  std::unique_ptr<TFE_TensorHandle, decltype(&TFE_DeleteTensorHandle)>
      value_handle(TFE_NewTensorHandle(t.get(), status),
                   TFE_DeleteTensorHandle);
  if (TF_GetCode(status) != TF_OK) return nullptr;

  TFE_OpAddInput(op, value_handle.get(), status);
  if (TF_GetCode(status) != TF_OK) return nullptr;

  num_retvals = 0;
  TFE_Execute(op, nullptr, &num_retvals, status);
  TFE_DeleteOp(op);
  if (TF_GetCode(status) != TF_OK) return nullptr;
  CHECK_EQ(0, num_retvals);

  TF_DeleteStatus(status);

  return var_handle;
}

TFE_Op* AddOp(TFE_Context* ctx, TFE_TensorHandle* a, TFE_TensorHandle* b) {
  TF_Status* status = TF_NewStatus();

  TFE_Op* op = TFE_NewOp(ctx, "AddV2", status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_OpAddInput(op, a, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_OpAddInput(op, b, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteStatus(status);
  TFE_OpSetAttrType(op, "T", TFE_TensorHandleDataType(a));

  return op;
}

TFE_Op* MatMulOp(TFE_Context* ctx, TFE_TensorHandle* a, TFE_TensorHandle* b) {
  TF_Status* status = TF_NewStatus();

  TFE_Op* op = TFE_NewOp(ctx, "MatMul", status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_OpAddInput(op, a, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_OpAddInput(op, b, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteStatus(status);
  TFE_OpSetAttrType(op, "T", TFE_TensorHandleDataType(a));

  return op;
}

TFE_Op* IdentityOp(TFE_Context* ctx, TFE_TensorHandle* a) {
  TF_Status* status = TF_NewStatus();

  TFE_Op* op = TFE_NewOp(ctx, "Identity", status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_OpAddInput(op, a, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteStatus(status);
  TFE_OpSetAttrType(op, "T", TFE_TensorHandleDataType(a));

  return op;
}

TFE_Op* ShapeOp(TFE_Context* ctx, TFE_TensorHandle* a) {
  TF_Status* status = TF_NewStatus();

  TFE_Op* op = TFE_NewOp(ctx, "Shape", status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_OpAddInput(op, a, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteStatus(status);
  TFE_OpSetAttrType(op, "T", TFE_TensorHandleDataType(a));

  return op;
}

TFE_TensorHandle* TestAxisTensorHandle(TFE_Context* ctx) {
  int64_t dims[] = {1};
  int data[] = {1};
  TF_Status* status = TF_NewStatus();
  TF_Tensor* t = TFE_AllocateHostTensor(ctx, TF_INT32, &dims[0],
                                        sizeof(dims) / sizeof(int64_t), status);
  memcpy(TF_TensorData(t), &data[0], TF_TensorByteSize(t));
  TFE_TensorHandle* th = TFE_NewTensorHandleFromTensor(ctx, t, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteTensor(t);
  TF_DeleteStatus(status);
  return th;
}

TFE_Op* MinOp(TFE_Context* ctx, TFE_TensorHandle* input,
              TFE_TensorHandle* axis) {
  TF_Status* status = TF_NewStatus();

  TFE_Op* op = TFE_NewOp(ctx, "Min", status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_OpAddInput(op, input, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_OpAddInput(op, axis, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_OpSetAttrBool(op, "keep_dims", 1);
  TFE_OpSetAttrType(op, "Tidx", TF_INT32);
  TF_DeleteStatus(status);
  TFE_OpSetAttrType(op, "T", TFE_TensorHandleDataType(input));

  return op;
}

TFE_Op* AllReduceOp(TFE_Context* ctx, TFE_TensorHandle* in, int group_size) {
  TF_Status* status = TF_NewStatus();

  TFE_Op* op = TFE_NewOp(ctx, "CollectiveReduce", status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_OpAddInput(op, in, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteStatus(status);

  TFE_OpSetAttrType(op, "T", TFE_TensorHandleDataType(in));
  TFE_OpSetAttrInt(op, "group_size", group_size);
  TFE_OpSetAttrInt(op, "group_key", 123);
  TFE_OpSetAttrInt(op, "instance_key", 456);
  TFE_OpSetAttrString(op, "merge_op", "Add", 3);
  TFE_OpSetAttrString(op, "final_op", "Id", 2);
  std::vector<int64_t> subdiv_offsets;
  TFE_OpSetAttrIntList(op, "subdiv_offsets", subdiv_offsets.data(),
                       subdiv_offsets.size());

  return op;
}

TFE_Op* SendOp(TFE_Context* ctx, TFE_TensorHandle* in,
               const std::string& op_name, const std::string& send_device,
               const std::string& recv_device,
               machina::uint64 send_device_incarnation) {
  TF_Status* status = TF_NewStatus();
  TFE_Op* op = TFE_NewOp(ctx, op_name.c_str(), status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_OpAddInput(op, in, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteStatus(status);

  TFE_OpSetAttrType(op, "T", TFE_TensorHandleDataType(in));
  TFE_OpSetAttrString(op, "tensor_name", "dummy", 5);
  TFE_OpSetAttrString(op, "send_device", send_device.c_str(),
                      send_device.size());
  TFE_OpSetAttrString(op, "recv_device", recv_device.c_str(),
                      recv_device.size());
  TFE_OpSetAttrInt(op, "send_device_incarnation", send_device_incarnation);

  return op;
}

TFE_Op* RecvOp(TFE_Context* ctx, const std::string& op_name,
               const std::string& send_device, const std::string& recv_device,
               machina::uint64 send_device_incarnation) {
  TF_Status* status = TF_NewStatus();
  TFE_Op* op = TFE_NewOp(ctx, op_name.c_str(), status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteStatus(status);

  TFE_OpSetAttrType(op, "tensor_type", TF_INT32);
  TFE_OpSetAttrString(op, "tensor_name", "dummy", 5);
  TFE_OpSetAttrString(op, "send_device", send_device.c_str(),
                      send_device.size());
  TFE_OpSetAttrString(op, "recv_device", recv_device.c_str(),
                      recv_device.size());
  TFE_OpSetAttrInt(op, "send_device_incarnation", send_device_incarnation);

  return op;
}

bool GetDeviceName(TFE_Context* ctx, string* device_name,
                   const char* device_type) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  TF_DeviceList* devices = TFE_ContextListDevices(ctx, status.get());
  CHECK_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  const int num_devices = TF_DeviceListCount(devices);
  for (int i = 0; i < num_devices; ++i) {
    const string dev_type(TF_DeviceListType(devices, i, status.get()));
    CHECK_EQ(TF_GetCode(status.get()), TF_OK) << TF_Message(status.get());
    const string dev_name(TF_DeviceListName(devices, i, status.get()));
    CHECK_EQ(TF_GetCode(status.get()), TF_OK) << TF_Message(status.get());
    if (dev_type == device_type) {
      *device_name = dev_name;
      LOG(INFO) << "Found " << device_type << " device " << *device_name;
      TF_DeleteDeviceList(devices);
      return true;
    }
  }
  TF_DeleteDeviceList(devices);
  return false;
}

machina::ServerDef GetServerDef(const string& job_name, int num_tasks) {
  machina::ServerDef server_def;
  server_def.set_protocol("grpc");
  server_def.set_job_name(job_name);
  server_def.set_task_index(0);
  machina::ClusterDef* cluster_def = server_def.mutable_cluster();
  machina::JobDef* job_def = cluster_def->add_job();
  job_def->set_name(job_name);
  for (int i = 0; i < num_tasks; i++) {
    int port = machina::testing::PickUnusedPortOrDie();
    job_def->mutable_tasks()->insert(
        {i, machina::strings::StrCat("localhost:", port)});
    LOG(INFO) << "Picked test port: " << port << " for job: " << job_name
              << ", task: " << i;
  }
  return server_def;
}

machina::ServerDef GetServerDef(int num_tasks) {
  return GetServerDef("localhost", num_tasks);
}

machina::ServerDef GetMultiClientServerDef(const std::string& job_name,
                                              int num_tasks,
                                              int num_virtual_gpus) {
  machina::ServerDef server_def;
  server_def.set_protocol("grpc");
  server_def.set_job_name(job_name);
  server_def.set_task_index(0);
  machina::ClusterDef* cluster_def = server_def.mutable_cluster();
  machina::JobDef* job_def = cluster_def->add_job();
  job_def->set_name(job_name);
  for (int i = 0; i < num_tasks; i++) {
    int port = machina::testing::PickUnusedPortOrDie();
    job_def->mutable_tasks()->insert(
        {i, machina::strings::StrCat("localhost:", port)});
  }
  auto* config = server_def.mutable_default_session_config();
  config->mutable_experimental()->set_collective_group_leader(
      machina::strings::StrCat("/job:", job_name, "/replica:0/task:", 0));
  auto* rewrite_options =
      config->mutable_graph_options()->mutable_rewrite_options();
  rewrite_options->set_scoped_allocator_optimization(
      machina::RewriterConfig::ON);
  rewrite_options->mutable_scoped_allocator_opts()->add_enable_op(
      "CollectiveReduce");

  if ((machina::IsGoogleCudaEnabled() || machina::IsBuiltWithROCm()) &&
      num_virtual_gpus > 0) {
    machina::GPUOptions* gpu_options =
        server_def.mutable_default_session_config()->mutable_gpu_options();
    auto virtual_devices =
        gpu_options->mutable_experimental()->add_virtual_devices();
    for (int i = 0; i < num_virtual_gpus; ++i) {
      virtual_devices->add_memory_limit_mb(200);
    }
  }
  return server_def;
}

TFE_TensorHandle* CreateVarHandle(TFE_Context* ctx,
                                  const machina::string& device_name,
                                  const machina::string& variable_name) {
  TF_Status* status = TF_NewStatus();
  // Create the variable handle.
  TFE_Op* op = TFE_NewOp(ctx, "VarHandleOp", status);
  if (TF_GetCode(status) != TF_OK) return nullptr;
  TFE_OpSetAttrType(op, "dtype", TF_FLOAT);
  TFE_OpSetAttrShape(op, "shape", {}, 0, status);
  TFE_OpSetAttrString(op, "container", "localhost", 0);
  TFE_OpSetAttrString(op, "shared_name", variable_name.data(),
                      variable_name.size());
  if (!device_name.empty()) {
    TFE_OpSetDevice(op, device_name.c_str(), status);
  }
  if (TF_GetCode(status) != TF_OK) return nullptr;
  TFE_TensorHandle* var_handle = nullptr;
  int num_retvals = 1;
  TFE_Execute(op, &var_handle, &num_retvals, status);
  if (TF_GetCode(status) != TF_OK) return nullptr;
  TFE_DeleteOp(op);
  if (TF_GetCode(status) != TF_OK) return nullptr;
  CHECK_EQ(1, num_retvals);
  TF_DeleteStatus(status);
  return var_handle;
}

TFE_TensorHandle* CreateVariable(TFE_Context* ctx, float value,
                                 const machina::string& device_name,
                                 const machina::string& variable_name) {
  TF_Status* status = TF_NewStatus();
  TFE_TensorHandle* var_handle =
      CreateVarHandle(ctx, device_name, variable_name);

  // Assign 'value' to it.
  TFE_Op* op = TFE_NewOp(ctx, "AssignVariableOp", status);
  if (TF_GetCode(status) != TF_OK) return nullptr;
  TFE_OpSetAttrType(op, "dtype", TF_FLOAT);
  TFE_OpAddInput(op, var_handle, status);
  if (!device_name.empty()) {
    TFE_OpSetDevice(op, device_name.c_str(), status);
  }

  // Convert 'value' to a TF_Tensor then a TFE_TensorHandle.
  std::unique_ptr<TF_Tensor, decltype(&TF_DeleteTensor)> t(
      TF_AllocateTensor(TF_FLOAT, nullptr, 0, sizeof(value)), TF_DeleteTensor);
  memcpy(TF_TensorData(t.get()), &value, TF_TensorByteSize(t.get()));

  std::unique_ptr<TFE_TensorHandle, decltype(&TFE_DeleteTensorHandle)>
      value_handle(TFE_NewTensorHandle(t.get(), status),
                   TFE_DeleteTensorHandle);
  if (TF_GetCode(status) != TF_OK) return nullptr;

  TFE_OpAddInput(op, value_handle.get(), status);
  if (TF_GetCode(status) != TF_OK) return nullptr;

  int num_retvals = 0;
  TFE_Execute(op, nullptr, &num_retvals, status);
  TFE_DeleteOp(op);
  if (TF_GetCode(status) != TF_OK) return nullptr;
  CHECK_EQ(0, num_retvals);
  TF_DeleteStatus(status);
  return var_handle;
}

TFE_Context* CreateContext(const std::string& serialized_server_def,
                           bool isolate_session_state,
                           int64_t init_timeout_in_ms) {
  TF_Status* status = TF_NewStatus();
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  opts->session_options.options.config.set_isolate_session_state(
      isolate_session_state);
  TFE_ContextOptionsSetAsync(opts, static_cast<unsigned char>(false));
  TFE_ContextOptionsSetDevicePlacementPolicy(opts, TFE_DEVICE_PLACEMENT_SILENT);
  TFE_Context* ctx = TFE_NewContext(opts, status);
  EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);
  TFE_ContextSetServerDefWithTimeout(ctx, 0, serialized_server_def.data(),
                                     serialized_server_def.size(),
                                     init_timeout_in_ms, status,
                                     /*clear_existing_contexts=*/false);
  EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);
  TFE_DeleteContextOptions(opts);
  TF_DeleteStatus(status);
  return ctx;
}

machina::ServerDef ReplaceTaskInServerDef(
    const machina::ServerDef& server_def, int task_index) {
  machina::ServerDef server_def_copy = server_def;
  machina::ClusterDef* cluster_def = server_def_copy.mutable_cluster();
  machina::JobDef* job_def = cluster_def->mutable_job(0);
  const int port = machina::testing::PickUnusedPortOrDie();
  job_def->mutable_tasks()->at(task_index) =
      machina::strings::StrCat("localhost:", port);
  return server_def_copy;
}

void ReplaceTaskInServerDef(machina::ServerDef* server_def, int task_index,
                            const string& host, int port) {
  machina::JobDef* job_def = server_def->mutable_cluster()->mutable_job(0);
  job_def->mutable_tasks()->at(task_index) =
      machina::strings::StrCat(host, ":", port);
}

std::vector<std::string> ListDeviceNames(TFE_Context* ctx) {
  TF_Status* status = TF_NewStatus();
  std::vector<std::string> device_names;
  TF_DeviceList* devices = TFE_ContextListDevices(ctx, status);
  EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);
  const int num_devices = TF_DeviceListCount(devices);
  for (int i = 0; i < num_devices; ++i) {
    device_names.emplace_back(TF_DeviceListName(devices, i, status));
    EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);
  }
  TF_DeleteDeviceList(devices);
  TF_DeleteStatus(status);
  return device_names;
}
