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

#ifndef MACHINA_LITE_DELEGATES_XNNPACK_UNARY_ELEMENTWISE_TESTER_H_
#define MACHINA_LITE_DELEGATES_XNNPACK_UNARY_ELEMENTWISE_TESTER_H_

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include "machina/lite/core/c/common.h"
#include "machina/lite/schema/schema_generated.h"

namespace tflite {
namespace xnnpack {

struct ToleranceInfo {
  float relative = 10.0f;
  float absolute = 0.0f;
};

class UnaryElementwiseTester {
 public:
  UnaryElementwiseTester() = default;
  UnaryElementwiseTester(const UnaryElementwiseTester&) = delete;
  UnaryElementwiseTester& operator=(const UnaryElementwiseTester&) = delete;

  inline UnaryElementwiseTester& Shape(std::initializer_list<int32_t> shape) {
    for (auto it = shape.begin(); it != shape.end(); ++it) {
      EXPECT_GT(*it, 0);
    }
    shape_ = std::vector<int32_t>(shape.begin(), shape.end());
    size_ = UnaryElementwiseTester::ComputeSize(shape_);
    return *this;
  }

  const std::vector<int32_t>& Shape() const { return shape_; }

  int32_t Size() const { return size_; }

  inline UnaryElementwiseTester& Tolerance(const ToleranceInfo& tolerance) {
    tolerance_ = tolerance;
    return *this;
  }

  const ToleranceInfo& Tolerance() const { return tolerance_; }
  float RelativeTolerance() const { return tolerance_.relative; }
  float AbsoluteTolerance() const { return tolerance_.absolute; }

  void Test(tflite::BuiltinOperator unary_op, TfLiteDelegate* delegate) const;

 private:
  std::vector<char> CreateTfLiteModel(tflite::BuiltinOperator unary_op) const;

  static int32_t ComputeSize(const std::vector<int32_t>& shape);

  std::vector<int32_t> shape_;
  int32_t size_;
  ToleranceInfo tolerance_;
};

}  // namespace xnnpack
}  // namespace tflite

#endif  // MACHINA_LITE_DELEGATES_XNNPACK_UNARY_ELEMENTWISE_TESTER_H_
