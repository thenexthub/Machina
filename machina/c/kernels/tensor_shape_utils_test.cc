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

#include "machina/c/kernels/tensor_shape_utils.h"

#include "machina/c/tf_tensor_internal.h"
#include "machina/core/framework/tensor.h"
#include "machina/core/framework/tensor_shape.h"
#include "machina/core/lib/core/status.h"
#include "machina/core/platform/test.h"
#include "machina/core/platform/types.h"

namespace machina {

namespace {

// A wrapper that will automatically delete the allocated TF_Tensor
// once out of scope.
struct TF_TensorWrapper {
  TF_Tensor* tf_tensor;
  explicit TF_TensorWrapper(TF_Tensor* tensor) { tf_tensor = tensor; }
  ~TF_TensorWrapper() { TF_DeleteTensor(tf_tensor); }
};

void TestShapeMatch(TensorShape shape) {
  Tensor tensor(DT_FLOAT, shape);
  absl::Status status;
  TF_Tensor* tf_tensor = TF_TensorFromTensor(tensor, &status);
  TF_TensorWrapper tensor_wrapper = TF_TensorWrapper(tf_tensor);
  ASSERT_TRUE(status.ok()) << status.ToString();
  ASSERT_EQ(tensor.shape().DebugString(), ShapeDebugString(tf_tensor));
}

TEST(ShapeDebugString, RegularShape) { TestShapeMatch(TensorShape({5, 4, 7})); }

TEST(ShapeDebugString, ScalarShape) { TestShapeMatch(TensorShape({})); }

}  // namespace
}  // namespace machina
