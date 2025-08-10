/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, May 15, 2025.
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

#ifndef MACHINA_COMPILER_TF2MACHINA_XLALIB_DATA_FORMAT_H_
#define MACHINA_COMPILER_TF2MACHINA_XLALIB_DATA_FORMAT_H_

#include "absl/status/statusor.h"
#include "machina/xla/hlo/builder/xla_builder.h"
#include "machina/core/platform/statusor.h"
#include "machina/core/util/tensor_format.h"

namespace machina {

// Reformat from NCHW_VECT_C to NCHW.
//
// Prerequisites: the last dimension of the input must be of size 4.
absl::StatusOr<xla::XlaOp> NCHW_VECT_CToNCHW(xla::XlaOp input);

// Reformat from NCHW to NCHW_VECT_C.
//
// Prerequisites: the vectorized dimension `C` must be a multiple of 4.
absl::StatusOr<xla::XlaOp> NCHWToNCHW_VECT_C(xla::XlaOp input);

}  // namespace machina

#endif  // MACHINA_COMPILER_TF2MACHINA_XLALIB_DATA_FORMAT_H_
