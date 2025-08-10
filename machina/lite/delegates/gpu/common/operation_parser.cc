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

#include "machina/lite/delegates/gpu/common/operation_parser.h"

#include <cstdint>

#include "absl/strings/str_cat.h"
#include "machina/lite/core/c/common.h"
#include "machina/lite/delegates/gpu/common/shape.h"
#include "machina/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {

absl::Status CheckMaxSupportedOpVersion(const TfLiteRegistration* registration,
                                        int max_version) {
  const int op_version = registration->version;
  if (op_version > max_version) {
    return absl::UnimplementedError(
        absl::StrCat("Max version supported: ", max_version,
                     ". Requested version ", op_version, "."));
  }
  return absl::OkStatus();
}

HW ToHW(int32_t h, int32_t w) { return HW(h > 0 ? h : 1, w > 0 ? w : 1); }

absl::Status ParsePoolingAttributes(const TfLitePoolParams* tf_options,
                                    const BHWC& input_shape,
                                    Pooling2DAttributes* attr) {
  attr->kernel = ToHW(tf_options->filter_height, tf_options->filter_width);
  attr->strides = ToHW(tf_options->stride_height, tf_options->stride_width);
  UpdatePadding(tf_options->padding, input_shape, attr);
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace tflite
