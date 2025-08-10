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
#ifndef MACHINA_LITE_DELEGATES_FLEX_KERNEL_H_
#define MACHINA_LITE_DELEGATES_FLEX_KERNEL_H_

#include <memory>

#include "machina/core/platform/status.h"
#include "machina/core/tfrt/fallback/op_kernel_runner.h"
#include "machina/lite/core/c/common.h"
#include "machina/lite/delegates/utils/simple_delegate.h"

namespace tflite {
namespace flex {

namespace testing {
class KernelTest;  // friend class declaration.
}  // namespace testing

struct OpData;
struct OpNode;

class DelegateKernel : public SimpleDelegateKernelInterface {
 public:
  DelegateKernel();
  ~DelegateKernel() override;

  TfLiteStatus Init(TfLiteContext* context,
                    const TfLiteDelegateParams* params) override;
  TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) override;
  TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) override;

 private:
  friend class tflite::flex::testing::KernelTest;

  // Validates that the computed output tensor shape for the Flex node matches
  // the existing output shape assigned to the output tensor.
  TfLiteStatus ValidateOutputTensorShapeConsistency(
      TfLiteContext* context) const;

  // Executes the Tensorflow op based on the inputs/outputs/attributes
  // information represented in the `node_data`.
  absl::Status ExecuteOpKernelRunner(
      machina::tfrt_stub::OpKernelRunState* run_state,
      TfLiteContext* context, OpNode* node_data);

  // Returns the tensor release map held in `op_data_`;
  const std::map<int, int>& GetTensorReleaseMap() const;

  std::unique_ptr<OpData> op_data_;

  // Indicates that the output shapes may be inferred using the input shapes and
  // May be allocated during Prepare.
  bool shapes_are_valid_ = true;
};

}  // namespace flex
}  // namespace tflite

#endif  // MACHINA_LITE_DELEGATES_FLEX_KERNEL_H_
