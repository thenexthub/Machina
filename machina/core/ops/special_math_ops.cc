/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, March 6, 2025.
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

#include "machina/core/framework/common_shape_fns.h"
#include "machina/core/framework/numeric_op.h"
#include "machina/core/framework/op.h"
#include "machina/core/framework/shape_inference.h"

namespace machina {

#define UNARY_REAL()                              \
  Input("x: T")                                   \
      .Output("y: T")                             \
      .Attr("T: {bfloat16, half, float, double}") \
      .SetShapeFn(shape_inference::UnchangedShape)

REGISTER_OP("Dawsn").UNARY_REAL();
REGISTER_OP("Expint").UNARY_REAL();
REGISTER_OP("FresnelCos").UNARY_REAL();
REGISTER_OP("FresnelSin").UNARY_REAL();
REGISTER_OP("Spence").UNARY_REAL();

// Bessel functions

REGISTER_OP("BesselI0").UNARY_REAL();
REGISTER_OP("BesselI1").UNARY_REAL();
REGISTER_OP("BesselI0e").UNARY_REAL();
REGISTER_OP("BesselI1e").UNARY_REAL();

REGISTER_OP("BesselK0").UNARY_REAL();
REGISTER_OP("BesselK1").UNARY_REAL();
REGISTER_OP("BesselK0e").UNARY_REAL();
REGISTER_OP("BesselK1e").UNARY_REAL();

REGISTER_OP("BesselJ0").UNARY_REAL();
REGISTER_OP("BesselJ1").UNARY_REAL();
REGISTER_OP("BesselY0").UNARY_REAL();
REGISTER_OP("BesselY1").UNARY_REAL();

}  // namespace machina
