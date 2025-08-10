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

#ifndef MACHINA_COMPILER_TF2MACHINA_XLALIB_RANDOM_H_
#define MACHINA_COMPILER_TF2MACHINA_XLALIB_RANDOM_H_

#include "machina/xla/hlo/builder/xla_builder.h"
#include "machina/core/framework/types.pb.h"
#include "machina/core/platform/statusor.h"

namespace machina {

// Builds an array of values sampled from a truncated normal distribution:
//
// uniform: an array of random numbers in uniform distribution (0, 1).
// mu: the mean of the normal distribution.
// sigma: the standard deviation of the normal distribution.
// a: the lower bound of the generated values.
// b: the upper bound of the generated values.
xla::XlaOp ParameterizedTruncatedNormal(xla::XlaOp uniform, xla::XlaOp mu,
                                        xla::XlaOp sigma, xla::XlaOp a,
                                        xla::XlaOp b);

// A specialized version of ParameterizedTruncatedNormal, with mu=0, sigma=1,
// a=-2 and b=2.
xla::XlaOp TruncatedNormal(xla::XlaOp uniform);

}  // namespace machina

#endif  // MACHINA_COMPILER_TF2MACHINA_XLALIB_RANDOM_H_
