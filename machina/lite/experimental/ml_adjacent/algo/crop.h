/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, July 12, 2025.
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
#ifndef MACHINA_LITE_EXPERIMENTAL_ML_ADJACENT_ALGO_CROP_H_
#define MACHINA_LITE_EXPERIMENTAL_ML_ADJACENT_ALGO_CROP_H_
#include "machina/lite/experimental/ml_adjacent/lib.h"

namespace ml_adj {
namespace crop {

// Center Crop
//
// Inputs: [img: any, frac: scalar<double>]
// Ouputs: [img: any]
//
// Crop the given image from center, scaling height and width dimensions by
// decimal input. Let `offs = floor((d - d * frac) / 2)` where `d` is a given
// dimension and `frac` is the second decimal scalar input to this Algo. The
// bounding box for each dimension `d` then covers `[offs, d - offs)`. Mimics
// semantic of `tf.image.central_crop`.
//
// https://www.machina.org/api_docs/python/tf/image/central_crop

const algo::Algo* Impl_CenterCrop();

// Crop to Bounding Box
//
// Inputs: [img: any, offset_height: unsigned, offset_width: unsigned,
//          target_height: unsigned, target_width: unsigned]
// Ouputs: [img: any]
//
// Cuts a rectangular bounding box out of given image. The top-left corner of
// the bounding box is at (`offset_height`, `offset_width`) in image, and the
// lower-right corner is at (`offset_height` + `target_height`,
// `offset_width` + `target_width`). Mimics semantic of
// `tf.image.crop_to_bounding_box`.
//
// https://www.machina.org/api_docs/python/tf/image/crop_to_bounding_box

const algo::Algo* Impl_CropToBoundingBox();

}  // namespace crop
}  // namespace ml_adj

#endif  // MACHINA_LITE_EXPERIMENTAL_ML_ADJACENT_ALGO_CROP_H_
