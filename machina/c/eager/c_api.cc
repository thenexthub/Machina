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

#include "machina/c/eager/c_api.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/memory/memory.h"
#include "machina/c/c_api.h"
#include "machina/c/c_api_internal.h"
#include "machina/c/eager/abstract_tensor_handle.h"
#include "machina/c/eager/c_api_experimental.h"
#include "machina/c/eager/c_api_internal.h"
#include "machina/c/eager/immediate_execution_operation.h"
#include "machina/c/eager/immediate_execution_tensor_handle.h"
#include "machina/c/eager/tfe_context_internal.h"
#include "machina/c/eager/tfe_op_internal.h"
#include "machina/c/eager/tfe_tensorhandle_internal.h"
#include "machina/c/tf_buffer_internal.h"
#include "machina/c/tf_status.h"
#include "machina/c/tf_tensor_internal.h"
#include "machina/xla/tsl/c/tsl_status_internal.h"
#include "machina/core/common_runtime/copy_tensor.h"
#include "machina/core/common_runtime/device.h"
#include "machina/core/common_runtime/device_factory.h"
#include "machina/core/common_runtime/device_mgr.h"
#include "machina/core/common_runtime/eager/attr_builder.h"
#include "machina/core/common_runtime/eager/context.h"
#include "machina/core/common_runtime/eager/custom_device.h"
#include "machina/core/common_runtime/eager/custom_device_op_handler.h"
#include "machina/core/common_runtime/eager/execute.h"
#include "machina/core/common_runtime/eager/placement_utils.h"
#include "machina/core/common_runtime/eager/tensor_handle.h"
#include "machina/core/common_runtime/function.h"
#include "machina/core/framework/attr_value.pb.h"
#include "machina/core/framework/device_attributes.pb.h"
#include "machina/core/framework/function.h"
#include "machina/core/framework/node_def_util.h"
#include "machina/core/framework/rendezvous.h"
#include "machina/core/framework/tensor_shape.pb.h"
#include "machina/core/framework/types.h"
#include "machina/core/platform/casts.h"
#include "machina/core/platform/errors.h"
#include "machina/core/platform/platform.h"
#include "machina/core/platform/status.h"
#include "machina/core/profiler/lib/traceme.h"
#include "machina/core/protobuf/error_codes.pb.h"
#include "machina/core/public/version.h"

#if !defined(IS_MOBILE_PLATFORM)
#include "machina/core/common_runtime/eager/context_distributed_manager.h"
#endif  // !IS_MOBILE_PLATFORM

using machina::string;

namespace {

string DeviceName(const machina::Device* d) {
  return (d == nullptr) ? "cpu:0" : d->name();
}

// Annotate eager runtime construction context to the given `function_def` as
// an attribute.
void AnnotateEagerRuntimeConstructionContext(
    machina::FunctionDef& function_def) {
  machina::AttrValue value;
  SetAttrValue("kEagerRuntime", &value);
  (*function_def.mutable_attr())["_construction_context"] = value;
}

}  // namespace

extern "C" {

TFE_ContextOptions* TFE_NewContextOptions() { return new TFE_ContextOptions; }

void TFE_ContextOptionsSetConfig(TFE_ContextOptions* options, const void* proto,
                                 size_t proto_len, TF_Status* status) {
  TF_SetConfig(&options->session_options, proto, proto_len, status);
}

void TFE_ContextOptionsSetAsync(TFE_ContextOptions* options,
                                unsigned char enable) {
  options->async = enable;
}

void TFE_ContextOptionsSetDevicePlacementPolicy(
    TFE_ContextOptions* options, TFE_ContextDevicePlacementPolicy policy) {
  options->device_placement_policy = policy;
}

void TFE_DeleteContextOptions(TFE_ContextOptions* options) { delete options; }

TFE_Context* TFE_NewContext(const TFE_ContextOptions* opts, TF_Status* status) {
  if (opts->use_tfrt) {
    status->status = machina::errors::Unimplemented("TFRT is not supported");
    return nullptr;
  }
  std::vector<std::unique_ptr<machina::Device>> devices;
  status->status = machina::DeviceFactory::AddDevices(
      opts->session_options.options, "/job:localhost/replica:0/task:0",
      &devices);
  if (!status->status.ok()) return nullptr;
  std::unique_ptr<machina::DeviceMgr> device_mgr(
      new machina::DynamicDeviceMgr(std::move(devices)));

  auto r = tsl::core::RefCountPtr<machina::IntraProcessRendezvous>(
      new machina::IntraProcessRendezvous(device_mgr.get()));
  machina::EagerContext* eager_context = new machina::EagerContext(
      opts->session_options.options,
      static_cast<machina::ContextDevicePlacementPolicy>(
          opts->device_placement_policy),
      opts->async, device_mgr.release(),
      /*device_mgr_owned*/ true, std::move(r),
      /*cluster_flr=*/nullptr,
      /*collective_executor_mgr=*/nullptr,
      /*run_eager_op_as_function=*/opts->run_eager_op_as_function,
      /*jit_compile_rewrite=*/opts->jit_compile_rewrite);
#if !defined(IS_MOBILE_PLATFORM)
  eager_context->SetDistributedManager(
      std::make_unique<machina::EagerContextDistributedManager>(
          eager_context));
#endif  // !IS_MOBILE_PLATFORM
  return machina::wrap(eager_context);
}

void TFE_DeleteContext(TFE_Context* ctx) {
  if (ctx == nullptr) {
    return;
  }

  // ctx->RefCountIsOne() should be true here.
  machina::unwrap(ctx)->Release();
}

TF_DeviceList* TFE_ContextListDevices(TFE_Context* ctx, TF_Status* status) {
  TF_DeviceList* l = new TF_DeviceList;
  machina::unwrap(ctx)->ListDevices(&l->response);
  return l;
}

void TFE_ContextClearCaches(TFE_Context* ctx) {
  machina::unwrap(ctx)->ClearCachesAndThreadExecutors();
}

// Set server_def on the context, possibly updating it.
TF_CAPI_EXPORT extern void TFE_ContextSetServerDef(TFE_Context* ctx,
                                                   int keep_alive_secs,
                                                   const void* proto,
                                                   size_t proto_len,
                                                   TF_Status* status) {
  TFE_ContextSetServerDefWithTimeoutAndRetries(
      ctx, keep_alive_secs, proto, proto_len, /*init_timeout_in_ms=*/0,
      /*retries=*/0, status, /*clear_existing_contexts=*/false);
}

// Set server def with timeout.
TF_CAPI_EXPORT extern void TFE_ContextSetServerDefWithTimeout(
    TFE_Context* ctx, int keep_alive_secs, const void* proto, size_t proto_len,
    int64_t init_timeout_in_ms, TF_Status* status,
    bool clear_existing_contexts) {
  TFE_ContextSetServerDefWithTimeoutAndRetries(
      ctx, keep_alive_secs, proto, proto_len, init_timeout_in_ms,
      /*retries=*/0, status, clear_existing_contexts);
}

// Set server_def on the context, possibly updating it.
// TODO(b/291142876) Simplify TFE_ContextSetServerDefWithTimeoutAndRetries and
// TFE_ContextUpdateServerDefWithTimeout to be simple wrappers around the same
// C++ function.
// Retries are used for CreateContext calls, which is used in
// ParameterServerStrategy initialization to be robust to worker preemption.
TF_CAPI_EXPORT extern void TFE_ContextSetServerDefWithTimeoutAndRetries(
    TFE_Context* ctx, int keep_alive_secs, const void* proto, size_t proto_len,
    int64_t init_timeout_in_ms, int retries, TF_Status* status,
    bool clear_existing_contexts) {
#if defined(IS_MOBILE_PLATFORM)
  status->status = machina::errors::Unimplemented(
      "TFE_ContextSetServerDef not supported on mobile");
#else   // !defined(IS_MOBILE_PLATFORM)
  machina::ServerDef server_def;
  if (!server_def.ParseFromArray(proto, proto_len)) {
    status->status = machina::errors::InvalidArgument(
        "Invalid machina.ServerDef protocol buffer");
    return;
  }
  status->status =
      machina::unwrap(ctx)->GetDistributedManager()->SetOrUpdateServerDef(
          server_def, /*reset_context=*/true, keep_alive_secs,
          init_timeout_in_ms, retries, clear_existing_contexts);
#endif  // !IS_MOBILE_PLATFORM
}

TF_CAPI_EXPORT extern void TFE_ContextUpdateServerDef(TFE_Context* ctx,
                                                      int keep_alive_secs,
                                                      const void* proto,
                                                      size_t proto_len,
                                                      TF_Status* status) {
  TFE_ContextUpdateServerDefWithTimeout(ctx, keep_alive_secs, proto, proto_len,
                                        /*init_timeout_in_ms=*/0, status);
}

TF_CAPI_EXPORT extern void TFE_ContextUpdateServerDefWithTimeout(
    TFE_Context* ctx, int keep_alive_secs, const void* proto, size_t proto_len,
    int64_t init_timeout_in_ms, TF_Status* status) {
#if defined(IS_MOBILE_PLATFORM)
  status->status = machina::errors::Unimplemented(
      "TFE_ContextUpdateServerDef not supported on mobile");
#else   // !defined(IS_MOBILE_PLATFORM)
  machina::ServerDef server_def;
  machina::EagerContext* context =
      machina::ContextFromInterface(machina::unwrap(ctx));
  if (!server_def.ParseFromArray(proto, proto_len)) {
    status->status = machina::errors::InvalidArgument(
        "Invalid machina.ServerDef protocol buffer");
    return;
  } else if (context->GetContextId() ==
             machina::EagerContext::kInvalidContextId) {
    status->status = machina::errors::InvalidArgument(
        "Trying to update a context with invalid context id.");
  }
  status->status =
      machina::unwrap(ctx)->GetDistributedManager()->SetOrUpdateServerDef(
          server_def, /*reset_context=*/false, keep_alive_secs,
          init_timeout_in_ms, /*retries=*/0);
#endif  // !IS_MOBILE_PLATFORM
}

TF_CAPI_EXPORT extern bool TFE_ContextCheckAlive(TFE_Context* ctx,
                                                 const char* worker_name,
                                                 TF_Status* status) {
#if defined(IS_MOBILE_PLATFORM)
  status->status = machina::errors::Unimplemented(
      "TFE_ContextSetServerDef not supported on mobile");
  return false;
#else   // !defined(IS_MOBILE_PLATFORM)
  bool is_alive;
  status->status =
      machina::unwrap(ctx)->GetDistributedManager()->CheckRemoteAlive(
          worker_name, &is_alive);
  return is_alive;
#endif  // !IS_MOBILE_PLATFORM
}

TF_CAPI_EXPORT extern void TFE_ContextAsyncWait(TFE_Context* ctx,
                                                TF_Status* status) {
#if defined(IS_MOBILE_PLATFORM)
  status->status = machina::OkStatus();
#else   // !defined(IS_MOBILE_PLATFORM)
  status->status = machina::unwrap(ctx)->AsyncWait();
#endif  // !IS_MOBILE_PLATFORM
}

void TFE_ContextSetThreadLocalDevicePlacementPolicy(
    TFE_Context* ctx, TFE_ContextDevicePlacementPolicy policy) {
  machina::unwrap(ctx)->SetThreadLocalDevicePlacementPolicy(
      static_cast<machina::ContextDevicePlacementPolicy>(policy));
}

// Note: this function looks up a thread local policy. So it should be called in
// the appropriate client thread. In particular, in async mode, it may not be
// safe to call this function from the async EagerExecutor threads.
extern TFE_ContextDevicePlacementPolicy TFE_ContextGetDevicePlacementPolicy(
    TFE_Context* ctx) {
  return static_cast<TFE_ContextDevicePlacementPolicy>(
      machina::unwrap(ctx)->GetDevicePlacementPolicy());
}

TFE_TensorHandle* TFE_NewTensorHandle(const TF_Tensor* t, TF_Status* status) {
  machina::Tensor tensor;
  status->status = machina::TF_TensorToTensor(t, &tensor);
  if (!status->status.ok()) return nullptr;

  return machina::wrap(machina::TensorHandle::CreateLocalHandle(tensor));
}

void TFE_DeleteTensorHandle(TFE_TensorHandle* h) {
  if (h == nullptr) return;

  tsl::profiler::TraceMe activity("TFE_DeleteTensorHandle",
                                  tsl::profiler::TraceMeLevel::kInfo);
  if (h) {
    machina::unwrap(h)->Unref();
  }
}

TF_DataType TFE_TensorHandleDataType(TFE_TensorHandle* h) {
  return static_cast<TF_DataType>(machina::unwrap(h)->DataType());
}

int TFE_TensorHandleNumDims(TFE_TensorHandle* h, TF_Status* status) {
  if (h == nullptr) {
    status->status = machina::errors::InvalidArgument("Invalid handle");
    return -1;
  }

  int num_dims = -1;
  status->status = machina::unwrap(h)->NumDims(&num_dims);
  return num_dims;
}

int64_t TFE_TensorHandleNumElements(TFE_TensorHandle* h, TF_Status* status) {
  if (h == nullptr) {
    status->status = machina::errors::InvalidArgument("Invalid handle");
    return -1;
  }

  int64_t num_elements = -1;
  status->status = machina::unwrap(h)->NumElements(&num_elements);
  return num_elements;
}

int64_t TFE_TensorHandleDim(TFE_TensorHandle* h, int dim_index,
                            TF_Status* status) {
  if (h == nullptr) {
    status->status = machina::errors::InvalidArgument("Invalid handle");
    return -1;
  }

  int64_t dim = -1;
  status->status = machina::unwrap(h)->Dim(dim_index, &dim);
  return dim;
}

const char* TFE_TensorHandleDeviceName(TFE_TensorHandle* h, TF_Status* status) {
  if (h == nullptr) {
    status->status = machina::errors::InvalidArgument("Invalid handle");
    return nullptr;
  }
  return machina::unwrap(h)->DeviceName(&status->status);
}

const char* TFE_TensorHandleBackingDeviceName(TFE_TensorHandle* h,
                                              TF_Status* status) {
  if (h == nullptr) {
    status->status = machina::errors::InvalidArgument("Invalid handle");
    return nullptr;
  }
  return machina::unwrap(h)->BackingDeviceName(&status->status);
}

TF_CAPI_EXPORT extern TFE_TensorHandle* TFE_TensorHandleCopySharingTensor(
    TFE_TensorHandle* h, TF_Status* status) {
  if (h == nullptr) {
    status->status = machina::errors::InvalidArgument("Invalid handle");
    return nullptr;
  }

  machina::unwrap(h)->Ref();
  return h;
}

TF_Tensor* TFE_TensorHandleResolve(TFE_TensorHandle* h, TF_Status* status) {
  if (h == nullptr) {
    status->status = machina::errors::InvalidArgument("Invalid handle");
    return nullptr;
  }

  machina::AbstractTensorInterface* t =
      machina::unwrap(h)->Resolve(&status->status);
  if (t == nullptr) {
    return nullptr;
  }

  return new TF_Tensor{t};
}

void* TFE_TensorHandleDevicePointer(TFE_TensorHandle* h, TF_Status* status) {
  if (h == nullptr) {
    status->status = machina::errors::InvalidArgument("Invalid handle");
    return nullptr;
  }
  machina::ImmediateExecutionTensorHandle* unwrapped_handle =
      machina::unwrap(h);
  // TODO(b/175427838): It would be nice to be able to use machina::isa here.
  if (machina::CustomDeviceTensorHandle::classof(unwrapped_handle)) {
    return machina::down_cast<machina::CustomDeviceTensorHandle*>(
               unwrapped_handle)
        ->DevicePointer();
  }
  // TODO(b/175427838): It would be nice to be able to use machina::isa here.
  if (!machina::TensorHandle::classof(unwrapped_handle)) {
    status->status = machina::errors::InvalidArgument("Invalid handle");
    return nullptr;
  }
  machina::TensorHandle* handle =
      machina::TensorHandleFromInterface(unwrapped_handle);

  if (handle->Type() != machina::TensorHandle::LOCAL) {
    status->status = machina::errors::InvalidArgument(
        "TFE_TensorHandleDevicePointer may not be called on a ",
        handle->TypeString(), " tensor handle.");
    return nullptr;
  }
  machina::Device* device(handle->device());
  if (device != nullptr) {
    status->status = device->Sync();
    if (!status->status.ok()) {
      return nullptr;
    }
  }
  const machina::Tensor* tensor;
  status->status = handle->Tensor(&tensor);
  if (!status->status.ok()) {
    return nullptr;
  }
  return const_cast<void*>(
      static_cast<const void*>(tensor->tensor_data().data()));
}

namespace machina {
namespace {
class CustomDeviceAPI : public machina::CustomDevice {
 public:
  CustomDeviceAPI(TFE_Context* context, TFE_CustomDevice device, void* info,
                  string name)
      : context_(context), device_(device), info_(info), name_(name) {}

  ~CustomDeviceAPI() override { device_.delete_device(info_); }

  const string& name() override { return name_; }

  absl::Status CopyTensorToDevice(
      ImmediateExecutionTensorHandle* handle,
      ImmediateExecutionTensorHandle** result) override {
    handle->Ref();
    TF_Status status;
    TFE_TensorHandle* result_handle = device_.copy_tensor_to_device(
        context_, machina::wrap(handle), &status, info_);
    handle->Unref();
    if (!status.status.ok()) return status.status;
    *result = machina::unwrap(result_handle);
    (*result)->Ref();
    TFE_DeleteTensorHandle(result_handle);
    return status.status;
  }

  absl::Status CopyTensorFromDevice(
      ImmediateExecutionTensorHandle* handle,
      const machina::string& target_device_name,
      ImmediateExecutionTensorHandle** result) override {
    TF_Status status;
    handle->Ref();
    TFE_TensorHandle* result_handle = device_.copy_tensor_from_device(
        context_, machina::wrap(handle), target_device_name.c_str(), &status,
        info_);
    handle->Unref();
    if (!status.status.ok()) return status.status;
    *result = machina::unwrap(result_handle);
    (*result)->Ref();
    TFE_DeleteTensorHandle(result_handle);
    return status.status;
  }

  absl::Status Execute(const ImmediateExecutionOperation* op,
                       ImmediateExecutionTensorHandle** retvals,
                       int* num_retvals) override {
    std::vector<TFE_TensorHandle*> outputs(*num_retvals);
    TF_Status status;
    device_.execute(machina::wrap(op), num_retvals, outputs.data(), &status,
                    info_);
    if (status.status.ok()) {
      for (int i = 0; i < *num_retvals; ++i) {
        retvals[i] = machina::unwrap(outputs[i]);
        retvals[i]->Ref();
        TFE_DeleteTensorHandle(outputs[i]);
      }
    }
    return status.status;
  }

  absl::Status Pack(absl::Span<ImmediateExecutionTensorHandle*> handles,
                    ImmediateExecutionTensorHandle** result) override {
    TF_Status status;
    *result = machina::unwrap(device_.pack(context_,
                                              machina::wrap(handles.data()),
                                              handles.size(), &status, info_));
    return status.status;
  }

  absl::StatusOr<bool> ShallPinToThisDevice(
      const ImmediateExecutionOperation* op) override {
    TF_Status status;
    // Let this custom device choose the device to pin this op on if it
    // implements the pinning function.
    if (device_.shall_pin_to_this_device != nullptr) {
      return device_.shall_pin_to_this_device(machina::wrap(op), &status);
    }
    return errors::Unimplemented("No custom device pinning implementation.");
  }

 private:
  TFE_Context* context_;
  TFE_CustomDevice device_;
  void* info_;
  string name_;
};

// An adapter which wraps the shape/data produced by C custom devices and uses
// it to implement custom device methods.
class CAPICustomDeviceTensorHandle
    : public machina::CustomDeviceTensorHandle {
 public:
  CAPICustomDeviceTensorHandle(machina::ImmediateExecutionContext* context,
                               machina::CustomDevice* device,
                               machina::DataType dtype, void* data,
                               TFE_CustomDeviceTensorHandleMethods methods)
      : machina::CustomDeviceTensorHandle(context, device, dtype),
        data_(data),
        methods_(methods) {}

  ~CAPICustomDeviceTensorHandle() override { methods_.deallocator(data_); }
  void* DevicePointer() const override { return data_; }
  absl::Status NumDims(int* num_dims) const override {
    TF_Status s;
    *num_dims = methods_.num_dims(data_, &s);
    return s.status;
  }
  absl::Status Dim(int dim_index, int64_t* dim) const override {
    TF_Status s;
    *dim = methods_.dim(data_, dim_index, &s);
    return s.status;
  }

  bool PreferCustomSummarizer() const override {
    return methods_.summarize != nullptr;
  }

  absl::Status SummarizeValue(std::string& summary) const override {
    if (methods_.summarize == nullptr) {
      return machina::CustomDeviceTensorHandle::SummarizeValue(summary);
    }
    TF_Status c_status;
    std::unique_ptr<TF_Buffer, decltype(&TF_DeleteBuffer)> summary_buffer(
        methods_.summarize(data_, &c_status), TF_DeleteBuffer);
    if (!c_status.status.ok()) {
      return c_status.status;
    }
    summary = std::string(reinterpret_cast<const char*>(summary_buffer->data),
                          summary_buffer->length);
    return absl::OkStatus();
  }

 private:
  void* const data_;
  const TFE_CustomDeviceTensorHandleMethods methods_;
};

}  // namespace
}  // namespace machina

TFE_TensorHandle* TFE_NewCustomDeviceTensorHandle(
    TFE_Context* ctx, const char* device_name, TF_DataType dtype, void* data,
    TFE_CustomDeviceTensorHandleMethods methods, TF_Status* status) {
  machina::ImmediateExecutionContext* context = machina::unwrap(ctx);
  machina::CustomDevice* device = nullptr;
  if (!context->GetCustomDeviceOpHandler().FindCustomDeviceFromName(device_name,
                                                                    &device)) {
    methods.deallocator(data);
    status->status =
        machina::errors::InvalidArgument(device_name, " unknown device.");
    return nullptr;
  }
  return machina::wrap(new machina::CAPICustomDeviceTensorHandle(
      context, device, *reinterpret_cast<machina::DataType*>(&dtype), data,
      methods));
}

TFE_TensorHandle* TFE_NewTensorHandleFromDeviceMemory(
    TFE_Context* ctx, const char* device_name, TF_DataType dtype,
    const int64_t* dims, int num_dims, void* data, size_t len,
    void (*deallocator)(void* data, size_t len, void* arg),
    void* deallocator_arg, TF_Status* status) {
  machina::Device* device = nullptr;
  machina::EagerContext* context =
      machina::ContextFromInterface(machina::unwrap(ctx));
  status->status = context->FindDeviceFromName(device_name, &device);
  if (!status->status.ok()) {
    deallocator(data, len, deallocator_arg);
    status->status =
        machina::errors::InvalidArgument(device_name, " unknown device.");
    return nullptr;
  }
  std::vector<int64_t> dimvec(num_dims);
  for (int i = 0; i < num_dims; ++i) {
    dimvec[i] = static_cast<int64_t>(dims[i]);
  }

  // TODO(apassos) do we need to wrap the deallocator here to make sure to sync
  // the device?
  TF_ManagedBuffer* buf =
      new TF_ManagedBuffer(data, len, deallocator, deallocator_arg,
                           /*owns_memory=*/false);

  machina::Tensor t(static_cast<machina::DataType>(dtype),
                       machina::TensorShape(dimvec), buf);
  buf->Unref();
  return machina::wrap(machina::TensorHandle::CreateLocalHandle(
      std::move(t), device, device, context));
}

// This function will block till the operation that produces `h` has
// completed. This is only valid on local TFE_TensorHandles. Returns the size in
// bytes of the memory pointed to by the device pointer returned above.
size_t TFE_TensorHandleDeviceMemorySize(TFE_TensorHandle* h,
                                        TF_Status* status) {
  if (h == nullptr) {
    status->status = machina::errors::InvalidArgument("Invalid handle");
    return 0;
  }
  machina::TensorHandle* handle =
      machina::TensorHandleFromInterface(machina::unwrap(h));
  if (handle->Type() != machina::TensorHandle::LOCAL) {
    status->status = machina::errors::InvalidArgument(
        "TFE_TensorHandleDeviceMemorySize may not be called on a ",
        handle->TypeString(), " tensor handle.");
    return 0;
  }
  const machina::Tensor* tensor;
  status->status = handle->Tensor(&tensor);
  if (!status->status.ok()) {
    return 0;
  }
  return tensor->TotalBytes();
}

TFE_Op* TFE_NewOp(TFE_Context* ctx, const char* op_or_function_name,
                  TF_Status* status) {
  machina::ImmediateExecutionOperation* new_op =
      machina::unwrap(ctx)->CreateOperation();
  status->status = new_op->Reset(op_or_function_name, nullptr);
  if (!status->status.ok()) {
    new_op->Release();
    new_op = nullptr;
  }
  return machina::wrap(new_op);
}

void TFE_DeleteOp(TFE_Op* op) {
  if (op == nullptr) {
    return;
  }

  machina::unwrap(op)->Release();
}

const char* TFE_OpGetName(const TFE_Op* op, TF_Status* status) {
  return machina::unwrap(op)->Name().c_str();
}

TFE_Context* TFE_OpGetContext(const TFE_Op* op, TF_Status* status) {
  return machina::wrap(machina::unwrap(op)->GetContext());
}

void TFE_OpSetDevice(TFE_Op* op, const char* device_name, TF_Status* status) {
  status->status = machina::unwrap(op)->SetDeviceName(device_name);
}

const char* TFE_OpGetDevice(const TFE_Op* op, TF_Status* status) {
  return machina::unwrap(op)->DeviceName().c_str();
}

void TFE_OpAddInput(TFE_Op* op, TFE_TensorHandle* input, TF_Status* status) {
  status->status = machina::unwrap(op)->AddInput(machina::unwrap(input));
}

void TFE_OpAddInputList(TFE_Op* op, TFE_TensorHandle** inputs, int num_inputs,
                        TF_Status* status) {
  status->status = machina::unwrap(op)->AddInputList(
      {reinterpret_cast<machina::AbstractTensorHandle**>(
           machina::unwrap(inputs)),
       static_cast<size_t>(num_inputs)});
}

extern int TFE_OpGetFlatInputCount(const TFE_Op* op, TF_Status* status) {
  return machina::unwrap(op)->GetInputs().size();
}

extern TFE_TensorHandle* TFE_OpGetFlatInput(const TFE_Op* op, int index,
                                            TF_Status* status) {
  return machina::wrap(machina::unwrap(op)->GetInputs()[index]);
}

TF_AttrType TFE_OpGetAttrType(TFE_Op* op, const char* attr_name,
                              unsigned char* is_list, TF_Status* status) {
  TF_AttrType ret = TF_ATTR_INT;
  const machina::AttrTypeMap* attr_types_;
  bool is_function;
  status->status = machina::AttrTypeMapForOp(
      machina::unwrap(op)->Name().c_str(), &attr_types_, &is_function);
  if (!status->status.ok()) {
    return ret;
  }
  status->status =
      machina::AttrTypeByName(*attr_types_, attr_name, &ret, is_list);
  return ret;
}

TF_AttrType TFE_OpNameGetAttrType(TFE_Context* ctx,
                                  const char* op_or_function_name,
                                  const char* attr_name, unsigned char* is_list,
                                  TF_Status* status) {
  TF_AttrType ret;
  TFE_Op* op = TFE_NewOp(ctx, op_or_function_name, status);
  if (status->status.ok()) {
    ret = TFE_OpGetAttrType(op, attr_name, is_list, status);
  } else {
    ret = TF_ATTR_INT;  // Same dummy return as TFE_OpGetAttrType.
  }
  TFE_DeleteOp(op);
  return ret;
}

void TFE_OpSetAttrString(TFE_Op* op, const char* attr_name, const void* value,
                         size_t length) {
  auto s = machina::unwrap(op)->SetAttrString(
      attr_name, static_cast<const char*>(value), length);
  if (!s.ok()) {
    LOG(WARNING) << "Unable to set attribute: " << attr_name;
  }
}

void TFE_OpSetAttrInt(TFE_Op* op, const char* attr_name, int64_t value) {
  auto s = machina::unwrap(op)->SetAttrInt(attr_name, value);
  if (!s.ok()) {
    LOG(WARNING) << "Unable to set attribute: " << attr_name;
  }
}

void TFE_OpSetAttrFloat(TFE_Op* op, const char* attr_name, float value) {
  auto s = machina::unwrap(op)->SetAttrFloat(attr_name, value);
  if (!s.ok()) {
    LOG(WARNING) << "Unable to set attribute: " << attr_name;
  }
}

void TFE_OpSetAttrBool(TFE_Op* op, const char* attr_name, unsigned char value) {
  auto s = machina::unwrap(op)->SetAttrBool(attr_name,
                                               (value == 0) ? false : true);
  if (!s.ok()) {
    LOG(WARNING) << "Unable to set attribute: " << attr_name;
  }
}

void TFE_OpSetAttrType(TFE_Op* op, const char* attr_name, TF_DataType value) {
  auto s = machina::unwrap(op)->SetAttrType(
      attr_name, static_cast<machina::DataType>(value));
  if (!s.ok()) {
    LOG(WARNING) << "Unable to set attribute: " << attr_name;
  }
}

void TFE_OpSetAttrShape(TFE_Op* op, const char* attr_name, const int64_t* dims,
                        const int num_dims, TF_Status* out_status) {
  out_status->status =
      machina::unwrap(op)->SetAttrShape(attr_name, dims, num_dims);
}

void TFE_OpSetAttrFunction(TFE_Op* op, const char* attr_name,
                           const TFE_Op* value) {
  auto s = machina::unwrap(op)->SetAttrFunction(
      attr_name, machina::unwrap(const_cast<TFE_Op*>(value)));
  if (!s.ok()) {
    LOG(WARNING) << "Unable to set attribute: " << attr_name;
  }
}

void TFE_OpSetAttrFunctionName(TFE_Op* op, const char* attr_name,
                               const char* data, size_t length) {
  auto s = machina::unwrap(op)->SetAttrFunctionName(attr_name, data, length);
  if (!s.ok()) {
    LOG(WARNING) << "Unable to set attribute: " << attr_name;
  }
}

void TFE_OpSetAttrTensor(TFE_Op* op, const char* attr_name, TF_Tensor* tensor,
                         TF_Status* status) {
  machina::Tensor t;
  status->status = TF_TensorToTensor(tensor, &t);
  machina::TensorInterface interface(t);
  status->status = machina::unwrap(op)->SetAttrTensor(attr_name, &interface);
}

void TFE_OpSetAttrStringList(TFE_Op* op, const char* attr_name,
                             const void* const* values, const size_t* lengths,
                             int num_values) {
  auto s = machina::unwrap(op)->SetAttrStringList(attr_name, values, lengths,
                                                     num_values);
  if (!s.ok()) {
    LOG(WARNING) << "Unable to set attribute: " << attr_name;
  }
}

void TFE_OpSetAttrFloatList(TFE_Op* op, const char* attr_name,
                            const float* values, int num_values) {
  auto s =
      machina::unwrap(op)->SetAttrFloatList(attr_name, values, num_values);
  if (!s.ok()) {
    LOG(WARNING) << "Unable to set attribute: " << attr_name;
  }
}

void TFE_OpSetAttrIntList(TFE_Op* op, const char* attr_name,
                          const int64_t* values, int num_values) {
  auto s =
      machina::unwrap(op)->SetAttrIntList(attr_name, values, num_values);
  if (!s.ok()) {
    LOG(WARNING) << "Unable to set attribute: " << attr_name;
  }
}

void TFE_OpSetAttrTypeList(TFE_Op* op, const char* attr_name,
                           const TF_DataType* values, int num_values) {
  auto s = machina::unwrap(op)->SetAttrTypeList(
      attr_name, reinterpret_cast<const machina::DataType*>(values),
      num_values);
  if (!s.ok()) {
    LOG(WARNING) << "Unable to set attribute: " << attr_name;
  }
}

void TFE_OpSetAttrBoolList(TFE_Op* op, const char* attr_name,
                           const unsigned char* values, int num_values) {
  auto s =
      machina::unwrap(op)->SetAttrBoolList(attr_name, values, num_values);
  if (!s.ok()) {
    LOG(WARNING) << "Unable to set attribute: " << attr_name;
  }
}

void TFE_OpSetAttrShapeList(TFE_Op* op, const char* attr_name,
                            const int64_t** dims, const int* num_dims,
                            int num_values, TF_Status* out_status) {
  out_status->status = machina::unwrap(op)->SetAttrShapeList(
      attr_name, dims, num_dims, num_values);
}

void TFE_OpSetAttrFunctionList(TFE_Op* op, const char* attr_name,
                               const TFE_Op** value, int num_values) {
  auto s = machina::unwrap(op)->SetAttrFunctionList(
      attr_name, {reinterpret_cast<const machina::AbstractOperation**>(
                      machina::unwrap(value)),
                  static_cast<size_t>(num_values)});
  if (!s.ok()) {
    LOG(WARNING) << "Unable to set attribute: " << attr_name;
  }
}

void TFE_OpSetAttrValueProto(const TFE_Op* op, const char* attr_name,
                             const void* proto, size_t proto_len,
                             TF_Status* status) {
  machina::AttrValue attr_value;
  if (!attr_value.ParseFromArray(proto, proto_len)) {
    status->status =
        machina::errors::InvalidArgument("Unparseable AttrValue proto");
    return;
  }
  if (op == nullptr) {
    status->status = machina::errors::InvalidArgument(
        "Got a null or uninitialized `op` argument");
    return;
  }
  machina::EagerOperation* operation =
      OperationFromInterface(machina::unwrap(const_cast<TFE_Op*>(op)));
  operation->MutableAttrs()->Set(attr_name, attr_value);
}

TF_CAPI_EXPORT extern int TFE_OpGetInputLength(TFE_Op* op,
                                               const char* input_name,
                                               TF_Status* status) {
  int ret = -1;
  status->status = machina::unwrap(op)->InputLength(input_name, &ret);
  return ret;
}

TF_CAPI_EXPORT extern int TFE_OpGetOutputLength(TFE_Op* op,
                                                const char* output_name,
                                                TF_Status* status) {
  int ret = -1;
  status->status = machina::unwrap(op)->OutputLength(output_name, &ret);
  return ret;
}

void TFE_Execute(TFE_Op* op, TFE_TensorHandle** retvals, int* num_retvals,
                 TF_Status* status) {
  machina::ImmediateExecutionOperation* unwrapped_op =
      machina::unwrap(op);

  status->status =
      unwrapped_op->GetContext()->GetCustomDeviceOpHandler().Execute(
          unwrapped_op,
          reinterpret_cast<machina::ImmediateExecutionTensorHandle**>(
              retvals),
          num_retvals);
}

TFE_TensorHandle* TFE_TensorHandleCopyToDevice(TFE_TensorHandle* h,
                                               TFE_Context* ctx,
                                               const char* device_name,
                                               TF_Status* status) {
  if (h == nullptr) {
    status->status = machina::errors::InvalidArgument("Invalid handle");
    return nullptr;
  }

  machina::ImmediateExecutionContext* unwrapped_ctx =
      machina::unwrap(ctx);

  auto* result =
      unwrapped_ctx->GetCustomDeviceOpHandler().CopyTensorHandleToDevice(
          unwrapped_ctx, machina::unwrap(h), device_name, &status->status);

  if (status->status.ok()) {
    return machina::wrap(result);
  }
  return nullptr;
}

void TFE_ContextAddFunctionDef(TFE_Context* ctx,
                               const char* serialized_function_def, size_t size,
                               TF_Status* status) {
  machina::FunctionDef function_def;
  if (!function_def.ParseFromArray(serialized_function_def, size)) {
    status->status =
        machina::errors::InvalidArgument("Invalid FunctionDef proto");
    return;
  }

  AnnotateEagerRuntimeConstructionContext(function_def);
  status->status = machina::unwrap(ctx)->AddFunctionDef(function_def);
}

void TFE_ContextAddFunction(TFE_Context* ctx, TF_Function* function,
                            TF_Status* status) {
  auto fdef_or = function->record->mutable_fdef();
  if (!fdef_or.ok()) {
    status->status = fdef_or.status();
    return;
  }

  AnnotateEagerRuntimeConstructionContext(*fdef_or.value());
  status->status = machina::unwrap(ctx)->AddFunctionDefWithStackTraces(
      *fdef_or.value(), function->record->stack_traces());
}

TF_Function* TFE_ContextGetFunction(TFE_Context* ctx, const char* name,
                                    TF_Status* status) {
  machina::core::RefCountPtr<machina::FunctionRecord> record =
      machina::unwrap(ctx)->FindRecord(name);

  if (record == nullptr) {
    status->status = machina::errors::NotFound(
        "Unable to find Function with name: ", name);
    return nullptr;
  }

  TF_Function* result = new TF_Function();
  record->Ref();
  result->record = record.get();
  return result;
}

void TFE_ContextRemoveFunction(TFE_Context* ctx, const char* name,
                               TF_Status* status) {
  status->status = machina::unwrap(ctx)->RemoveFunction(name);
}

unsigned char TFE_ContextHasFunction(TFE_Context* ctx, const char* name) {
  return machina::unwrap(ctx)->FindFunctionDef(name) != nullptr;
}

void TFE_ContextEnableRunMetadata(TFE_Context* ctx) {
  machina::unwrap(ctx)->SetShouldStoreGraphs(true);
}

void TFE_ContextDisableRunMetadata(TFE_Context* ctx) {
  machina::unwrap(ctx)->SetShouldStoreGraphs(false);
}

}  // extern "C"

TFE_TensorHandle* TFE_NewTensorHandle(const machina::Tensor& t,
                                      TF_Status* status) {
  return machina::wrap(machina::TensorHandle::CreateLocalHandle(t));
}

void TFE_ContextExportRunMetadata(TFE_Context* ctx, TF_Buffer* buf,
                                  TF_Status* status) {
  auto* context = machina::unwrap(ctx);
  status->status = context->AsyncWait();
  if (!status->status.ok()) return;
  auto run_metadata = context->ExportRunMetadata();
  status->status = MessageToBuffer(*run_metadata, buf);
}

namespace {
TFE_Op* GetFunc(TFE_Context* ctx, const machina::NameAttrList& func,
                TF_Status* status) {
  TFE_Op* func_op = TFE_NewOp(ctx, func.name().data(), status);
  for (const auto& attr : func.attr()) {
    if (!status->status.ok()) return nullptr;
    SetOpAttrValueScalar(ctx, func_op, attr.second, attr.first.data(), status);
    if (!status->status.ok()) return nullptr;
  }
  return func_op;
}
}  // namespace

void TFE_ContextStartStep(TFE_Context* ctx) {
  machina::unwrap(ctx)->StartStep();
}

void TFE_ContextEndStep(TFE_Context* ctx) {
  machina::unwrap(ctx)->EndStep();
}

const TFE_OpAttrs* TFE_OpGetAttrs(const TFE_Op* op) {
  return machina::wrap(machina::unwrap(op)->GetOpAttrs());
}

void TFE_OpAddAttrs(TFE_Op* op, const TFE_OpAttrs* attrs) {
  machina::unwrap(op)->AddAttrs(machina::unwrap(attrs));
}

void TFE_OpAttrsSerialize(const TFE_OpAttrs* attrs, TF_Buffer* buf,
                          TF_Status* status) {
  machina::NameAttrList name_and_attrs;
  machina::unwrap(attrs)->GetNameAttrList(&name_and_attrs);
  status->status = MessageToBuffer(name_and_attrs, buf);
}

namespace machina {
void SetOpAttrValueScalar(TFE_Context* ctx, TFE_Op* op,
                          const machina::AttrValue& default_value,
                          const char* attr_name, TF_Status* status) {
  switch (default_value.value_case()) {
    case machina::AttrValue::kS: {
      const string& v = default_value.s();
      TFE_OpSetAttrString(op, attr_name, v.data(), v.size());
      break;
    }
    case machina::AttrValue::kI:
      TFE_OpSetAttrInt(op, attr_name, static_cast<int64_t>(default_value.i()));
      break;
    case machina::AttrValue::kF:
      TFE_OpSetAttrFloat(op, attr_name, default_value.f());
      break;
    case machina::AttrValue::kB:
      TFE_OpSetAttrBool(op, attr_name, default_value.b());
      break;
    case machina::AttrValue::kType:
      TFE_OpSetAttrType(op, attr_name,
                        static_cast<TF_DataType>(default_value.type()));
      break;
    case machina::AttrValue::kShape: {
      const auto& tensor_shape = default_value.shape();
      if (tensor_shape.unknown_rank()) {
        TFE_OpSetAttrShape(op, attr_name, nullptr, -1, status);
      } else {
        const auto num_dims = tensor_shape.dim_size();
        std::unique_ptr<int64_t[]> dims(new int64_t[num_dims]);
        for (int i = 0; i < num_dims; ++i) {
          dims[i] = tensor_shape.dim(i).size();
        }
        TFE_OpSetAttrShape(op, attr_name, dims.get(), num_dims, status);
      }
    } break;
    case machina::AttrValue::kFunc: {
      const auto func_op = GetFunc(ctx, default_value.func(), status);
      if (!status->status.ok()) return;
      // TODO(nareshmodi): TFE_OpSetAttrFunction and TFE_OpSetAttrFunctionList
      // require TFE_Op* and just convert it internally a NameAttrValue, so
      // consider adding an overload to the C API to make this case easier.
      TFE_OpSetAttrFunction(op, attr_name, func_op);
      TFE_DeleteOp(func_op);
    } break;
    case machina::AttrValue::kList: {
      // String
      if (const int s_size = default_value.list().s_size()) {
        absl::InlinedVector<const void*, 4> values_vector;
        values_vector.reserve(s_size);
        absl::InlinedVector<size_t, 4> lengths_vector;
        lengths_vector.reserve(s_size);
        for (int i = 0; i < s_size; ++i) {
          const string& v = default_value.list().s(i);
          values_vector.push_back(v.data());
          lengths_vector.push_back(v.size());
        }
        TFE_OpSetAttrStringList(op, attr_name, values_vector.data(),
                                lengths_vector.data(), s_size);
      }

      // Int
      if (const int i_size = default_value.list().i_size()) {
        absl::InlinedVector<int64_t, 4> i_vector;
        i_vector.reserve(i_size);
        for (int i = 0; i < i_size; ++i) {
          i_vector.push_back(default_value.list().i(i));
        }
        TFE_OpSetAttrIntList(op, attr_name, i_vector.data(), i_size);
      }
      // Float
      if (const int f_size = default_value.list().f_size()) {
        absl::InlinedVector<float, 4> f_vector;
        f_vector.reserve(f_size);
        for (int i = 0; i < f_size; ++i) {
          f_vector.push_back(default_value.list().f(i));
        }
        TFE_OpSetAttrFloatList(op, attr_name, f_vector.data(), f_size);
      }
      // Bool
      if (const int b_size = default_value.list().b_size()) {
        absl::InlinedVector<unsigned char, 4> b_vector;
        b_vector.reserve(b_size);
        for (int i = 0; i < b_size; i++) {
          b_vector.push_back(default_value.list().b(i));
        }
        TFE_OpSetAttrBoolList(op, attr_name, b_vector.data(), b_size);
      }
      // Type
      if (const int type_size = default_value.list().type_size()) {
        absl::InlinedVector<unsigned int, 4> type_vector;
        type_vector.reserve(type_size);
        for (int i = 0; i < type_size; ++i) {
          type_vector.push_back(default_value.list().type(i));
        }
        TFE_OpSetAttrTypeList(
            op, attr_name,
            reinterpret_cast<const TF_DataType*>(type_vector.data()),
            type_size);
      }

      // Rest are not supported.
      if (default_value.list().shape_size() > 0 ||
          default_value.list().func_size() > 0 ||
          default_value.list().tensor_size() > 0) {
        TF_SetStatus(
            status, TF_UNIMPLEMENTED,
            machina::strings::StrCat("Unable to get setfor default value: ",
                                        default_value.DebugString())
                .data());
      }
    } break;
    case machina::AttrValue::kTensor:
      TF_FALLTHROUGH_INTENDED;
    case machina::AttrValue::kPlaceholder:
      TF_FALLTHROUGH_INTENDED;
    case machina::AttrValue::VALUE_NOT_SET:
      TF_SetStatus(
          status, TF_UNIMPLEMENTED,
          machina::strings::StrCat("Unable to get setfor default value: ",
                                      default_value.DebugString())
              .data());
  }
}
}  // namespace machina

namespace {
TFE_TensorHandle* DefaultCustomDevicePack(TFE_Context* context,
                                          TFE_TensorHandle** handles,
                                          int num_handles, TF_Status* status,
                                          void* device_info) {
  TF_SetStatus(status, TF_UNIMPLEMENTED,
               "This custom device does not support packing tensors.");
  return nullptr;
}
}  // namespace

extern "C" {

bool TFE_IsCustomDevice(TFE_Context* ctx, const char* device_name) {
  return machina::unwrap(ctx)->IsCustomDevice(device_name);
}

void TFE_RegisterCustomDevice(TFE_Context* ctx, TFE_CustomDevice device,
                              const char* device_name, void* device_info,
                              TF_Status* status) {
  // Fill in default values for optional functionality.
  if (device.pack == nullptr) {
    device.pack = &DefaultCustomDevicePack;
  }
  auto custom_device = std::make_unique<machina::CustomDeviceAPI>(
      ctx, device, device_info, device_name);
  status->status = machina::unwrap(ctx)->RegisterCustomDevice(
      device_name, std::move(custom_device));
}

}  // extern "C"
