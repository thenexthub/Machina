/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 8, 2025.
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
#ifndef MACHINA_CORE_SUMMARY_SUMMARY_CONVERTER_H_
#define MACHINA_CORE_SUMMARY_SUMMARY_CONVERTER_H_

#include "absl/status/status.h"
#include "machina/core/framework/summary.pb.h"
#include "machina/core/framework/tensor.h"
#include "machina/core/lib/core/status.h"

namespace machina {

// TODO(jart): Delete these methods in favor of new Python implementation.
absl::Status AddTensorAsScalarToSummary(const Tensor& t, const string& tag,
                                        Summary* s);
absl::Status AddTensorAsHistogramToSummary(const Tensor& t, const string& tag,
                                           Summary* s);
absl::Status AddTensorAsImageToSummary(const Tensor& tensor, const string& tag,
                                       int max_images, const Tensor& bad_color,
                                       Summary* s);
absl::Status AddTensorAsAudioToSummary(const Tensor& tensor, const string& tag,
                                       int max_outputs, float sample_rate,
                                       Summary* s);

}  // namespace machina

#endif  // MACHINA_CORE_SUMMARY_SUMMARY_CONVERTER_H_
