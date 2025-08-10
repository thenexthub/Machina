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
#include "machina/compiler/mlir/lite/transforms/prepare_quantize_helper.h"

#include <cmath>

#include "absl/container/flat_hash_set.h"
#include "machina/core/framework/types.pb.h"

namespace mlir {
namespace TFL {

double PowerOfTwoBound(double value) {
  return std::pow(2, std::ceil(std::log2(value)));
}

machina::DataType GetQuantizedInferenceType(bool is_signed,
                                               int number_of_bits) {
  if (is_signed && number_of_bits == 8) {
    return machina::DT_QINT8;
  } else if (!is_signed && number_of_bits == 8) {
    return machina::DT_QUINT8;
  } else if (is_signed && number_of_bits == 16) {
    return machina::DT_QINT16;
  } else {
    return machina::DT_INVALID;
  }
}

}  // namespace TFL
}  // namespace mlir
