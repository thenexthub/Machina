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

#include <stdint.h>

#include "machina/c/c_api_macros.h"

#ifndef MACHINA_C_TF_SHAPE_H_
#define MACHINA_C_TF_SHAPE_H_

#ifdef __cplusplus
extern "C" {
#endif

// An opaque type corresponding to a shape in machina. In the future,
// we may expose the ABI of TF_Shape for performance reasons.
typedef struct TF_Shape TF_Shape;

// Return a new, unknown rank shape object. The caller is responsible for
// calling TF_DeleteShape to deallocate and destroy the returned shape.
TF_CAPI_EXPORT extern TF_Shape* TF_NewShape();

// Returns the rank of `shape`. If `shape` has unknown rank, returns -1.
TF_CAPI_EXPORT extern int TF_ShapeDims(const TF_Shape* shape);

// Returns the `d`th dimension of `shape`. If `shape` has unknown rank,
// invoking this function is undefined behavior. Returns -1 if dimension is
// unknown.
TF_CAPI_EXPORT extern int64_t TF_ShapeDimSize(const TF_Shape* shape, int d);

// Deletes `shape`.
TF_CAPI_EXPORT extern void TF_DeleteShape(TF_Shape* shape);

#ifdef __cplusplus
} /* end extern "C" */
#endif

#endif  // MACHINA_C_TF_SHAPE_H_
