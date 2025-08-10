
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

#include <sstream>
#include <string>

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "machina/c/kernels.h"
#include "machina/c/kernels/tensor_shape_utils.h"
#include "machina/c/tf_status.h"
#include "machina/c/tf_tensor.h"
#include "machina/core/framework/registration/registration.h"
#include "machina/core/framework/summary.pb.h"
#include "machina/core/framework/types.h"
#include "machina/core/platform/bfloat16.h"
#include "machina/core/platform/logging.h"
#include "machina/core/platform/macros.h"
#include "machina/core/platform/protobuf.h"
#include "machina/core/platform/strcat.h"
#include "machina/core/platform/tstring.h"
#include "machina/core/platform/types.h"

namespace {

// Struct that stores the status and TF_Tensor inputs to the opkernel.
// Used to delete tensor and status in its destructor upon kernel return.
struct Params {
  TF_Tensor* tags;
  TF_Tensor* values;
  TF_Status* status;
  explicit Params(TF_OpKernelContext* ctx)
      : tags(nullptr), values(nullptr), status(nullptr) {
    status = TF_NewStatus();
    TF_GetInput(ctx, 0, &tags, status);
    if (TF_GetCode(status) == TF_OK) {
      TF_GetInput(ctx, 1, &values, status);
    }
  }
  ~Params() {
    TF_DeleteStatus(status);
    TF_DeleteTensor(tags);
    TF_DeleteTensor(values);
  }
};

// dummy functions used for kernel registration
void* ScalarSummaryOp_Create(TF_OpKernelConstruction* ctx) { return nullptr; }

void ScalarSummaryOp_Delete(void* kernel) {}

// Helper functions for compute method
bool IsSameSize(TF_Tensor* tensor1, TF_Tensor* tensor2);
// Returns a string representation of a single tag or empty string if there
// are multiple tags
std::string SingleTag(TF_Tensor* tags);

template <typename T>
void ScalarSummaryOp_Compute(void* kernel, TF_OpKernelContext* ctx) {
  Params params(ctx);
  if (TF_GetCode(params.status) != TF_OK) {
    TF_OpKernelContext_Failure(ctx, params.status);
    return;
  }
  if (!IsSameSize(params.tags, params.values)) {
    std::ostringstream err;
    err << "tags and values are not the same shape: "
        << machina::ShapeDebugString(params.tags)
        << " != " << machina::ShapeDebugString(params.values)
        << SingleTag(params.tags);
    TF_SetStatus(params.status, TF_INVALID_ARGUMENT, err.str().c_str());
    TF_OpKernelContext_Failure(ctx, params.status);
    return;
  }
  // Convert tags and values tensor to array to access elements by index
  machina::Summary s;
  auto tags_array =
      static_cast<machina::tstring*>(TF_TensorData(params.tags));
  auto values_array = static_cast<T*>(TF_TensorData(params.values));
  // Copy tags and values into summary protobuf
  for (int i = 0; i < TF_TensorElementCount(params.tags); ++i) {
    machina::Summary::Value* v = s.add_value();
    const machina::tstring& Ttags_i = tags_array[i];
    v->set_tag(Ttags_i.data(), Ttags_i.size());
    v->set_simple_value(static_cast<float>(values_array[i]));
  }
  TF_Tensor* summary_tensor =
      TF_AllocateOutput(ctx, 0, TF_ExpectedOutputDataType(ctx, 0), nullptr, 0,
                        sizeof(machina::tstring), params.status);
  if (TF_GetCode(params.status) != TF_OK) {
    TF_DeleteTensor(summary_tensor);
    TF_OpKernelContext_Failure(ctx, params.status);
    return;
  }
  machina::tstring* output_tstring =
      reinterpret_cast<machina::tstring*>(TF_TensorData(summary_tensor));
  CHECK(SerializeToTString(s, output_tstring));
  TF_DeleteTensor(summary_tensor);
}

bool IsSameSize(TF_Tensor* tensor1, TF_Tensor* tensor2) {
  if (TF_NumDims(tensor1) != TF_NumDims(tensor2)) {
    return false;
  }
  for (int d = 0; d < TF_NumDims(tensor1); d++) {
    if (TF_Dim(tensor1, d) != TF_Dim(tensor2, d)) {
      return false;
    }
  }
  return true;
}

std::string SingleTag(TF_Tensor* tags) {
  if (TF_TensorElementCount(tags) == 1) {
    const char* single_tag =
        static_cast<machina::tstring*>(TF_TensorData(tags))->c_str();
    return machina::strings::StrCat(" (tag '", single_tag, "')");
  } else {
    return "";
  }
}

template <typename T>
void RegisterScalarSummaryOpKernel() {
  TF_Status* status = TF_NewStatus();
  {
    auto* builder = TF_NewKernelBuilder(
        "ScalarSummary", machina::DEVICE_CPU, &ScalarSummaryOp_Create,
        &ScalarSummaryOp_Compute<T>, &ScalarSummaryOp_Delete);
    TF_KernelBuilder_TypeConstraint(
        builder, "T",
        static_cast<TF_DataType>(machina::DataTypeToEnum<T>::v()), status);
    CHECK_EQ(TF_OK, TF_GetCode(status)) << "Error while adding type constraint";
    TF_RegisterKernelBuilder("ScalarSummary", builder, status);
    CHECK_EQ(TF_OK, TF_GetCode(status))
        << "Error while registering Scalar Summmary kernel";
  }
  TF_DeleteStatus(status);
}

// A dummy static variable initialized by a lambda whose side-effect is to
// register the ScalarSummary kernel.
TF_ATTRIBUTE_UNUSED bool IsScalarSummaryOpKernelRegistered = []() {
  if (SHOULD_REGISTER_OP_KERNEL("ScalarSummary")) {
    RegisterScalarSummaryOpKernel<int64_t>();
    RegisterScalarSummaryOpKernel<machina::uint64>();
    RegisterScalarSummaryOpKernel<machina::int32>();
    RegisterScalarSummaryOpKernel<machina::uint32>();
    RegisterScalarSummaryOpKernel<machina::uint16>();
    RegisterScalarSummaryOpKernel<machina::int16>();
    RegisterScalarSummaryOpKernel<machina::int8>();
    RegisterScalarSummaryOpKernel<machina::uint8>();
    RegisterScalarSummaryOpKernel<Eigen::half>();
    RegisterScalarSummaryOpKernel<machina::bfloat16>();
    RegisterScalarSummaryOpKernel<float>();
    RegisterScalarSummaryOpKernel<double>();
  }
  return true;
}();
}  // namespace
