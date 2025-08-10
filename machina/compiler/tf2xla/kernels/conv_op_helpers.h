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

#ifndef MACHINA_COMPILER_TF2MACHINA_MACHINA_XLA_KERNELS_CONV_OP_HELPERS_H_
#define MACHINA_COMPILER_TF2MACHINA_MACHINA_XLA_KERNELS_CONV_OP_HELPERS_H_

#include <cstdint>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "machina/xla/hlo/builder/xla_builder.h"
#include "machina/core/framework/op_kernel.h"
#include "machina/core/framework/types.h"
#include "machina/core/framework/types.pb.h"
#include "machina/core/platform/statusor.h"
#include "machina/core/util/padding.h"
#include "machina/core/util/tensor_format.h"

// This header exposes utilities for translating TensorFlow convolution ops into
// XLA ops.
//
// conv_ops.cc contains lowerings for many of these TF convolution ops (e.g.
// Conv2D, Conv3DBackpropFilterV2), but you might want to use the utilities in
// this header to implement a new and exciting convolution op, for example a
// fused TensorFlow op that contains a convolution and other things.

namespace machina {

// We don't support integers for convolutions for GPU, so we list the supported
// types for non-gpu and gpu here.
std::vector<DataType> GetXlaConvTypesForNonGpu();
std::vector<DataType> GetXlaConvTypesForGpu();

// ConvOpAttrs contains all of the metadata necessary to specify a TF or XLA
// convolution.
struct ConvOpAttrs {
  // Constructs a ConvOpAttrs, reading most of the attributes from `ctx`.
  static absl::StatusOr<ConvOpAttrs> Create(int num_spatial_dims,
                                            bool depthwise,
                                            OpKernelConstruction* ctx);

  bool depthwise;
  int num_spatial_dims;
  std::vector<int32> dilations;
  std::vector<int32> strides;
  Padding padding;
  std::vector<int64_t> explicit_paddings;
  TensorFormat data_format;
};

// Helper for the general Conv Op.
struct ConvNDOpAttrs {
  // Constructs a ConvOpAttrs, reading most of the attributes from `ctx`.
  static absl::StatusOr<ConvNDOpAttrs> Create(OpKernelConstruction* ctx);

  int groups;
  int batch_dims;
  std::vector<int32> dilations;
  std::vector<int32> strides;
  Padding padding;
  std::vector<int64_t> explicit_paddings;
  TensorFormat data_format;
};

// Creates a new XLA forward or backward convolution with the given inputs and
// attributes.
absl::StatusOr<xla::XlaOp> MakeXlaForwardConvOp(absl::string_view type_string,
                                                xla::XlaOp conv_input,
                                                xla::XlaOp filter,
                                                const ConvOpAttrs& attrs);
absl::StatusOr<xla::XlaOp> MakeXlaBackpropInputConvOp(
    absl::string_view type_string, const xla::Shape& input_shape,
    xla::XlaOp filter, xla::XlaOp out_backprop, const ConvOpAttrs& attrs,
    xla::XlaOp* input_sizes = nullptr);
absl::StatusOr<xla::XlaOp> MakeXlaBackpropFilterConvOp(
    absl::string_view type_string, xla::XlaOp activations,
    const xla::Shape& filter_shape, xla::XlaOp gradients,
    const ConvOpAttrs& attrs);

}  // namespace machina

#endif  // MACHINA_COMPILER_TF2MACHINA_MACHINA_XLA_KERNELS_CONV_OP_HELPERS_H_
