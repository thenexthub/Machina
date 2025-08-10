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
#ifndef MACHINA_LITE_DELEGATES_HEXAGON_BUILDERS_TESTS_HEXAGON_DELEGATE_OP_MODEL_H_
#define MACHINA_LITE_DELEGATES_HEXAGON_BUILDERS_TESTS_HEXAGON_DELEGATE_OP_MODEL_H_

#include <algorithm>

#include <gtest/gtest.h>
#include "machina/lite/core/c/common.h"
#include "machina/lite/core/kernels/register.h"
#include "machina/lite/core/model.h"
#include "machina/lite/delegates/hexagon/hexagon_delegate.h"
#include "machina/lite/interpreter.h"
#include "machina/lite/kernels/internal/reference/reference_ops.h"
#include "machina/lite/kernels/internal/tensor.h"
#include "machina/lite/kernels/internal/types.h"
#include "machina/lite/kernels/test_util.h"
#include "machina/lite/schema/schema_generated.h"

namespace tflite {
class SingleOpModelWithHexagon : public SingleOpModel {
 public:
  SingleOpModelWithHexagon() : delegate_(nullptr, [](TfLiteDelegate*) {}) {
    SetBypassDefaultDelegates();
  }

  void ApplyDelegateAndInvoke() {
    static const char kDelegateName[] = "TfLiteHexagonDelegate";

    // Make sure we set the environment.
    setenv(
        "ADSP_LIBRARY_PATH",
        "/data/local/tmp/hexagon_delegate_test;/system/lib/rfsa/adsp;/system/"
        "vendor/lib/rfsa/adsp;/dsp",
        1 /*overwrite*/);

    // For tests, we use one-op-models.
    params_.min_nodes_per_partition = 1;
    auto* delegate_ptr = TfLiteHexagonDelegateCreate(&params_);
    ASSERT_TRUE(delegate_ptr != nullptr);
    delegate_ = Interpreter::TfLiteDelegatePtr(
        delegate_ptr, [](TfLiteDelegate* delegate) {
          TfLiteHexagonDelegateDelete(delegate);
          // Turn off the fast rpc and cleanup.
          // Any communication with the DSP will fail unless new
          // HexagonDelegateInit called.
          TfLiteHexagonTearDown();
        });
    TfLiteHexagonInit();
    // Make sure we have valid interpreter.
    ASSERT_TRUE(interpreter_ != nullptr);
    // Add delegate.
    EXPECT_TRUE(interpreter_->ModifyGraphWithDelegate(delegate_.get()) !=
                kTfLiteError);
    // Make sure graph has one Op which is the delegate node.
    ASSERT_EQ(1, interpreter_->execution_plan().size());
    const int node = interpreter_->execution_plan()[0];
    const auto* node_and_reg = interpreter_->node_and_registration(node);
    ASSERT_TRUE(node_and_reg != nullptr);
    ASSERT_TRUE(node_and_reg->second.custom_name != nullptr);
    ASSERT_STREQ(kDelegateName, node_and_reg->second.custom_name);

    Invoke();
  }

 protected:
  using SingleOpModel::builder_;

 private:
  Interpreter::TfLiteDelegatePtr delegate_;
  TfLiteHexagonDelegateOptions params_ = {0};
};
}  // namespace tflite

#endif  // MACHINA_LITE_DELEGATES_HEXAGON_BUILDERS_TESTS_HEXAGON_DELEGATE_OP_MODEL_H_
