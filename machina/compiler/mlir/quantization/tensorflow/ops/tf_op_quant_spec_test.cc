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

#include "machina/compiler/mlir/quantization/machina/ops/tf_op_quant_spec.h"

#include <gtest/gtest.h>
#include "machina/compiler/mlir/quantization/machina/quantization_options.pb.h"

namespace mlir::quant {
namespace {

using QuantizationOptions = machina::quantization::QuantizationOptions;
using QuantizationComponentSpec =
    machina::quantization::QuantizationComponentSpec;

TEST(TfOpQuantSpecTest, WeightComponentSpecExist) {
  QuantizationOptions quant_options;
  QuantizationComponentSpec quant_spec;
  quant_spec.set_quantization_component(
      QuantizationComponentSpec::COMPONENT_WEIGHT);
  quant_spec.set_tensor_type(QuantizationComponentSpec::TENSORTYPE_INT_8);
  auto mutable_quant_method = quant_options.mutable_quantization_method();
  *mutable_quant_method->add_quantization_component_specs() = quant_spec;
  auto output = GetWeightComponentSpec(quant_options);
  EXPECT_TRUE(output.has_value());
}

TEST(TfOpQuantSpecTest, WeightComponentSpecDoNotExist) {
  QuantizationOptions quant_options;
  auto output = GetWeightComponentSpec(quant_options);
  EXPECT_FALSE(output.has_value());
}

}  // namespace
}  // namespace mlir::quant
