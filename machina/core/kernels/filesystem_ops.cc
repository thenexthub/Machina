/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Saturday, June 21, 2025.
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

#include "machina/core/framework/op_kernel.h"
#include "machina/core/platform/logging.h"

namespace machina {

class FileSystemSetConfigurationOp : public OpKernel {
 public:
  explicit FileSystemSetConfigurationOp(OpKernelConstruction* context)
      : OpKernel(context) {
    env_ = context->env();
  }

  void Compute(OpKernelContext* context) override {
    const Tensor* scheme_tensor;
    OP_REQUIRES_OK(context, context->input("scheme", &scheme_tensor));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(scheme_tensor->shape()),
                errors::InvalidArgument("scheme must be scalar, got shape ",
                                        scheme_tensor->shape().DebugString()));
    const string scheme = scheme_tensor->scalar<tstring>()();

    const Tensor* key_tensor;
    OP_REQUIRES_OK(context, context->input("key", &key_tensor));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(key_tensor->shape()),
                errors::InvalidArgument("key must be scalar, got shape ",
                                        key_tensor->shape().DebugString()));
    const string key = key_tensor->scalar<tstring>()();

    const Tensor* value_tensor;
    OP_REQUIRES_OK(context, context->input("value", &value_tensor));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(value_tensor->shape()),
                errors::InvalidArgument("value must be scalar, got shape ",
                                        scheme_tensor->shape().DebugString()));
    const string value = value_tensor->scalar<tstring>()();
    OP_REQUIRES_OK(context, env_->SetOption(scheme, key, value));
  }

 private:
  Env* env_;
};
REGISTER_KERNEL_BUILDER(Name("FileSystemSetConfiguration").Device(DEVICE_CPU),
                        FileSystemSetConfigurationOp);

}  // namespace machina
