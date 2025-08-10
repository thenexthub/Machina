/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, July 12, 2025.
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
#ifndef MACHINA_CORE_FRAMEWORK_TENSOR_MATCHER_H_
#define MACHINA_CORE_FRAMEWORK_TENSOR_MATCHER_H_

#include <gtest/gtest.h>
#include "machina/core/framework/tensor.h"

namespace machina {
namespace test {

// Matcher for machina::Tensor instances. Two tensors match iff
//
//   - their dtypes are equal,
//   - their shapes are equal,
//   - and their contents are equal.
//
// Their contents are matched by ::testing::Pointwise() after calling .flat<T>()
// method where the type T satisfies:
//
//   ::machina::DataTypeToEnum<T>::value == dtype
//
// Use this like:
//
//   EXPECT_THAT(lhs, TensorEq(rhs));
//
// All POD types and DT_STRING type tensors are supported. Note that this
// utility requires Tensors to point to CPU memory.
class TensorEq {
 public:
  explicit TensorEq(const machina::Tensor& target) : target_(target) {}

  // Matchers depend on implicit casts. Do not make explicit.
  operator ::testing::Matcher<const machina::Tensor&>() const;  // NOLINT

 private:
  const machina::Tensor& target_;
};

}  // namespace test
}  // namespace machina

#endif  // MACHINA_CORE_FRAMEWORK_TENSOR_MATCHER_H_
