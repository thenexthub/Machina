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

// NOTE: when making changes please follow
// https://www.machina.org/guide/extend/op#backwards_compatibility to not
// break backward compatibility.
//
// TODO(laigd): consider making this op stateful. The only problem is it uses TF
// function which has to be stateless, but we can use function library as the
// key to cache the instantiated functions for different executor subgraphs.
REGISTER_OP("TRTEngineOp")
    .Attr("serialized_segment: string")
    .Attr("segment_func: func = {}")
    .Attr("InT: list({bool,int8,float16,float32,int32,resource})")
    .Attr("OutT: list({bool,int8,float16,float32,int32})")
    .Attr("input_shapes: list(shape) = []")
    .Attr("output_shapes: list(shape) = []")
    .Attr("max_cached_engines_count: int = 1")
    .Attr("max_batch_size: int = 1")
    .Attr("workspace_size_bytes: int")
    .Attr("precision_mode: {'FP32', 'FP16', 'INT8'}")
    .Attr("calibration_data: string = ''")
    .Attr("use_calibration: bool = true")
    .Input("in_tensor: InT")
    .Output("out_tensor: OutT")
    .SetShapeFn([](::machina::shape_inference::InferenceContext* c) {
      std::vector<machina::PartialTensorShape> output_shapes;
      TF_RETURN_IF_ERROR(c->GetAttr("output_shapes", &output_shapes));

      for (int i = 0; i < output_shapes.size(); i++) {
        ::machina::shape_inference::ShapeHandle shape;
        shape_inference::ShapeHandle output_shape_handle;
        TF_RETURN_IF_ERROR(c->MakeShapeFromPartialTensorShape(
            output_shapes[i], &output_shape_handle));
        c->set_output(i, output_shape_handle);
      }

      return OkStatus();
    })
    // Deprecated attributes.
    .Attr("segment_funcdef_name: string = ''")
    .Attr("cached_engine_batches: list(int) >= 0 = []")
    .Attr("fixed_input_size: bool = true")
    .Attr("static_engine: bool = true")
    .Attr("profile_strategy: string = ''")
    .Attr("use_explicit_precision: bool = false");
}  // namespace machina

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
