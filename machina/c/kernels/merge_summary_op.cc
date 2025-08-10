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
#include <memory>
#include <sstream>
#include <unordered_set>

#include "absl/log/check.h"
#include "machina/c/kernels.h"
#include "machina/c/tf_status.h"
#include "machina/c/tf_tensor.h"
#include "machina/core/framework/registration/registration.h"
#include "machina/core/framework/summary.pb.h"
#include "machina/core/framework/types.h"
#include "machina/core/platform/logging.h"
#include "machina/core/platform/macros.h"
#include "machina/core/platform/protobuf.h"
#include "machina/core/platform/tstring.h"

namespace {

// Operators used to create a std::unique_ptr for TF_Tensor and TF_Status
struct TFTensorDeleter {
  void operator()(TF_Tensor* tf_tensor) const { TF_DeleteTensor(tf_tensor); }
};

struct TFStatusDeleter {
  void operator()(TF_Status* tf_status) const { TF_DeleteStatus(tf_status); }
};

// Struct that wraps TF_Tensor and TF_Status to delete once out of scope
using Safe_TF_TensorPtr = std::unique_ptr<TF_Tensor, TFTensorDeleter>;
using Safe_TF_StatusPtr = std::unique_ptr<TF_Status, TFStatusDeleter>;

// dummy functions used for kernel registration
void* MergeSummaryOp_Create(TF_OpKernelConstruction* ctx) { return nullptr; }

void MergeSummaryOp_Delete(void* kernel) {}

void MergeSummaryOp_Compute(void* kernel, TF_OpKernelContext* ctx) {
  machina::Summary s;
  std::unordered_set<machina::string> tags;
  Safe_TF_StatusPtr status(TF_NewStatus());
  for (int input_num = 0; input_num < TF_NumInputs(ctx); ++input_num) {
    TF_Tensor* input;
    TF_GetInput(ctx, input_num, &input, status.get());
    Safe_TF_TensorPtr safe_input_ptr(input);
    if (TF_GetCode(status.get()) != TF_OK) {
      TF_OpKernelContext_Failure(ctx, status.get());
      return;
    }
    auto tags_array =
        static_cast<machina::tstring*>(TF_TensorData(safe_input_ptr.get()));
    for (int i = 0; i < TF_TensorElementCount(safe_input_ptr.get()); ++i) {
      const machina::tstring& s_in = tags_array[i];
      machina::Summary summary_in;
      if (!machina::ParseProtoUnlimited(&summary_in, s_in)) {
        TF_SetStatus(status.get(), TF_INVALID_ARGUMENT,
                     "Could not parse one of the summary inputs");
        TF_OpKernelContext_Failure(ctx, status.get());
        return;
      }
      for (int v = 0; v < summary_in.value_size(); ++v) {
        // This tag is unused by the TensorSummary op, so no need to check for
        // duplicates.
        const machina::string& tag = summary_in.value(v).tag();
        if ((!tag.empty()) && !tags.insert(tag).second) {
          std::ostringstream err;
          err << "Duplicate tag " << tag << " found in summary inputs ";
          TF_SetStatus(status.get(), TF_INVALID_ARGUMENT, err.str().c_str());
          TF_OpKernelContext_Failure(ctx, status.get());
          return;
        }
        *s.add_value() = summary_in.value(v);
      }
    }
  }
  Safe_TF_TensorPtr summary_tensor(TF_AllocateOutput(
      /*context=*/ctx, /*index=*/0, /*dtype=*/TF_ExpectedOutputDataType(ctx, 0),
      /*dims=*/nullptr, /*num_dims=*/0,
      /*len=*/sizeof(machina::tstring), status.get()));
  if (TF_GetCode(status.get()) != TF_OK) {
    TF_OpKernelContext_Failure(ctx, status.get());
    return;
  }
  machina::tstring* output_tstring = reinterpret_cast<machina::tstring*>(
      TF_TensorData(summary_tensor.get()));
  CHECK(SerializeToTString(s, output_tstring));
}

void RegisterMergeSummaryOpKernel() {
  TF_Status* status = TF_NewStatus();
  {
    auto* builder = TF_NewKernelBuilder(
        "MergeSummary", machina::DEVICE_CPU, &MergeSummaryOp_Create,
        &MergeSummaryOp_Compute, &MergeSummaryOp_Delete);
    TF_RegisterKernelBuilder("MergeSummary", builder, status);
    CHECK_EQ(TF_OK, TF_GetCode(status))
        << "Error while registering Merge Summmary kernel";
  }
  TF_DeleteStatus(status);
}

// A dummy static variable initialized by a lambda whose side-effect is to
// register the Histogram Summary kernel.
TF_ATTRIBUTE_UNUSED static bool IsMergeSummaryOpKernelRegistered = []() {
  if (SHOULD_REGISTER_OP_KERNEL("MergeSummary")) {
    RegisterMergeSummaryOpKernel();
  }
  return true;
}();

}  // namespace
