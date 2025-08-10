/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, June 15, 2025.
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

#include "machina/lite/core/async/testing/test_backend.h"

#include <cstddef>
#include <string>

#include "machina/lite/array.h"
#include "machina/lite/builtin_ops.h"
#include "machina/lite/core/async/c/types.h"
#include "machina/lite/core/c/c_api_types.h"
#include "machina/lite/core/c/common.h"
#include "machina/lite/delegates/utils.h"

namespace tflite {
namespace async {
namespace testing {

namespace {

TfLiteStatus DelegatePrepare(TfLiteContext* context,
                             TfLiteDelegate* tflite_delegate) {
  auto* backend = reinterpret_cast<TestBackend*>(tflite_delegate->data_);

  // Can delegate all nodes.
  delegates::IsNodeSupportedFn node_supported_fn =
      [=](TfLiteContext* context, TfLiteNode* node,
          TfLiteRegistration* registration,
          std::string* unsupported_details) -> bool { return true; };

  delegates::GraphPartitionHelper helper(context, node_supported_fn);
  TF_LITE_ENSURE_STATUS(helper.Partition(nullptr));

  auto supported_nodes = helper.GetNodesOfFirstNLargestPartitions(
      backend->NumPartitions(), backend->MinPartitionedNodes());

  // Create TfLiteRegistration with the provided async kernel.
  TfLiteRegistration reg{};
  reg.init = [](TfLiteContext* context, const char* buffer,
                size_t length) -> void* {
    const TfLiteDelegateParams* params =
        reinterpret_cast<const TfLiteDelegateParams*>(buffer);
    auto* backend = reinterpret_cast<TestBackend*>(params->delegate->data_);
    // AsyncSubgraph requires TfLiteNode.user_data to be of TfLiteAsyncKernel
    // type.
    return backend->get_kernel();
  };
  reg.free = [](TfLiteContext*, void*) -> void {};
  reg.prepare = [](TfLiteContext*, TfLiteNode*) -> TfLiteStatus {
    return kTfLiteOk;
  };
  reg.invoke = [](TfLiteContext*, TfLiteNode*) -> TfLiteStatus {
    return kTfLiteOk;
  };
  reg.profiling_string = nullptr;
  reg.builtin_code = kTfLiteBuiltinDelegate;
  reg.custom_name = "TestBackend";
  reg.version = 1;
  reg.async_kernel = [](TfLiteContext*,
                        TfLiteNode* node) -> TfLiteAsyncKernel* {
    return reinterpret_cast<TfLiteAsyncKernel*>(node->user_data);
  };

  return context->ReplaceNodeSubsetsWithDelegateKernels(
      context, reg, BuildTfLiteArray(supported_nodes).get(), tflite_delegate);
}

}  // namespace

TestBackend::TestBackend(TfLiteAsyncKernel* kernel)
    : kernel_(kernel), delegate_(TfLiteDelegateCreate()) {
  delegate_.Prepare = &DelegatePrepare;
  delegate_.CopyFromBufferHandle = nullptr;
  delegate_.CopyToBufferHandle = nullptr;
  delegate_.FreeBufferHandle = nullptr;
  delegate_.data_ = this;
}

}  // namespace testing
}  // namespace async
}  // namespace tflite
