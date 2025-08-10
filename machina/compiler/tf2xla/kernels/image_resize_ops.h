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
#ifndef MACHINA_COMPILER_TF2MACHINA_XLAKERNELS_IMAGE_RESIZE_OPS_H_
#define MACHINA_COMPILER_TF2MACHINA_XLAKERNELS_IMAGE_RESIZE_OPS_H_

#include "machina/compiler/tf2xla/xla_op_kernel.h"
#include "machina/xla/primitive_util.h"
#include "machina/xla/xla_data.pb.h"
#include "machina/core/framework/op_kernel.h"

namespace machina {

class ResizeNearestNeighborOp : public XlaOpKernel {
 public:
  explicit ResizeNearestNeighborOp(OpKernelConstruction* ctx);
  void Compile(XlaOpKernelContext* ctx) override;

 protected:
  bool align_corners_ = true;
  bool half_pixel_centers_ = true;
  bool is_kernel_bilinear_ = false;
};

class ResizeBilinearOp : public XlaOpKernel {
 public:
  explicit ResizeBilinearOp(OpKernelConstruction* ctx);

  void Compile(XlaOpKernelContext* ctx) override;

 protected:
  bool align_corners_ = true;
  bool half_pixel_centers_ = true;
  bool is_kernel_bilinear_ = true;
};

class ResizeBilinearGradOp : public XlaOpKernel {
 public:
  explicit ResizeBilinearGradOp(OpKernelConstruction* ctx);

  void Compile(XlaOpKernelContext* ctx) override;

 protected:
  bool align_corners_;
  bool half_pixel_centers_ = true;
  xla::PrimitiveType output_type_;
};

}  // namespace machina

#endif  // MACHINA_COMPILER_TF2MACHINA_XLAKERNELS_IMAGE_RESIZE_OPS_H_
