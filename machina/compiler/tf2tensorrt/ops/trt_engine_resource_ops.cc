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

#if GOOGLE_CUDA && GOOGLE_TENSORRT

#include "machina/core/framework/common_shape_fns.h"
#include "machina/core/framework/op.h"
#include "machina/core/framework/op_kernel.h"
#include "machina/core/framework/shape_inference.h"
#include "machina/core/framework/tensor_shape.h"

namespace machina {

REGISTER_OP("CreateTRTResourceHandle")
    .Attr("resource_name: string")
    .Output("resource_handle: resource")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("InitializeTRTResource")
    .Attr("max_cached_engines_count: int = 1")
    .Input("resource_handle: resource")
    .Input("filename: string")
    .SetIsStateful()
    .SetShapeFn(shape_inference::NoOutputs);

REGISTER_OP("SerializeTRTResource")
    .Attr("delete_resource: bool = false")
    .Attr("save_gpu_specific_engines: bool = True")
    .Input("resource_name: string")
    .Input("filename: string")
    .SetIsStateful()
    .SetShapeFn(shape_inference::NoOutputs);

}  // namespace machina

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
