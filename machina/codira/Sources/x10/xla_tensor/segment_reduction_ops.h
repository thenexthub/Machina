/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Sunday, August 10, 2025.
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

#ifndef X10_XLA_TENSOR_SEGMENT_REDUCTION_OPS_H_
#define X10_XLA_TENSOR_SEGMENT_REDUCTION_OPS_H_

#include "machina/compiler/xla/client/xla_builder.h"

namespace codira_xla {

xla::XlaOp UnsortedSegmentReduce(
    xla::XlaOp data, xla::XlaOp indices, xla::XlaOp init_value,
    xla::int64 num_segments,
    const std::function<xla::XlaOp(xla::XlaOp, xla::XlaOp)>& combine);

}  // namespace codira_xla

#endif  // X10_XLA_TENSOR_SEGMENT_REDUCTION_OPS_H_
