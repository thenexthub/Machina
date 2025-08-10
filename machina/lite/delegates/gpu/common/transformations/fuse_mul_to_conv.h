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

#ifndef MACHINA_LITE_DELEGATES_GPU_COMMON_TRANSFORMATIONS_FUSE_MUL_TO_CONV_H_
#define MACHINA_LITE_DELEGATES_GPU_COMMON_TRANSFORMATIONS_FUSE_MUL_TO_CONV_H_

#include <memory>

#include "machina/lite/delegates/gpu/common/model_transformer.h"
#include "machina/lite/delegates/gpu/common/operations.h"

namespace tflite {
namespace gpu {

// Fuse Multiply Scalar or Multiply Broadcast after Convolution(Convolution2D,
// DepthWise, TransposedConvolution, FullyConnected) into weights and biases of
// convolution.
std::unique_ptr<SequenceTransformation> NewMergeConvolutionWithMul();

// Fuse Multiply Scalar or Multiply Broadcast before Convolution(Convolution2D,
// DepthWise, TransposedConvolution, FullyConnected) into weights and biases of
// convolution.
std::unique_ptr<SequenceTransformation> NewMergeMulWithConvolution();

// Modify Convolution2DAttributes so that after making convolution with
// modified attributes we will have the same result as convolution
// with old attributes and following multiply operation.
void FuseConvolution2DWithMultiply(const ElementwiseAttributes& mul_attr,
                                   Convolution2DAttributes* attr);

// Modify DepthwiseConvolution2DAttributes so that after making depth wise
// convolution with modified attributes we will have the same result as depth
// wise convolution with old attributes and following multiply operation.
void FuseDepthwiseConvolution2DWithMultiply(
    const ElementwiseAttributes& mul_attr,
    DepthwiseConvolution2DAttributes* attr);

// Modify ConvolutionTransposedAttributes so that after making convolution
// transposed with modified attributes we will have the same result as
// convolution transposed with old attributes and following multiply operation.
void FuseConvolutionTransposedWithMultiply(
    const ElementwiseAttributes& mul_attr,
    ConvolutionTransposedAttributes* attr);

// Modify FullyConnectedAttributes so that after making fully connected with
// modified attributes we will have the same result as fully connected
// with old attributes and following multiply operation.
void FuseFullyConnectedWithMultiply(const ElementwiseAttributes& mul_attr,
                                    FullyConnectedAttributes* attr);

// Modify Convolution2DAttributes so that after making convolution with
// modified attributes we will have the same result as multiply operation and
// convolution with old attributes
void FuseMultiplyWithConvolution2D(const ElementwiseAttributes& mul_attr,
                                   Convolution2DAttributes* attr);

// Modify DepthwiseConvolution2DAttributes so that after making depth wise
// convolution with modified attributes we will have the same result as multiply
// operation and depth wise convolution with old attributes
void FuseMultiplyWithDepthwiseConvolution2D(
    const ElementwiseAttributes& mul_attr,
    DepthwiseConvolution2DAttributes* attr);

// Modify ConvolutionTransposedAttributes so that after making convolution
// transposed with modified attributes we will have the same result as multiply
// operation and convolution transposed with old attributes
void FuseMultiplyWithConvolutionTransposed(
    const ElementwiseAttributes& mul_attr,
    ConvolutionTransposedAttributes* attr);

// Modify FullyConnectedAttributes so that after making fully connected
// with modified attributes we will have the same result as multiply
// operation and fully connected with old attributes
void FuseMultiplyWithFullyConnected(const ElementwiseAttributes& mul_attr,
                                    FullyConnectedAttributes* attr);

}  // namespace gpu
}  // namespace tflite

#endif  // MACHINA_LITE_DELEGATES_GPU_COMMON_TRANSFORMATIONS_FUSE_MUL_TO_CONV_H_
