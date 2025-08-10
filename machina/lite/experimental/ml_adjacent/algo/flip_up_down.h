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
#ifndef MACHINA_LITE_EXPERIMENTAL_ML_ADJACENT_ALGO_FLIP_UP_DOWN_H_
#define MACHINA_LITE_EXPERIMENTAL_ML_ADJACENT_ALGO_FLIP_UP_DOWN_H_

#include "machina/lite/experimental/ml_adjacent/lib.h"

namespace ml_adj {
namespace flip_up_down {

// Flip (up to down)
//
// Inputs: [img: any]
// Ouputs: [img: any]
//
// Flips the given image vertically (up to down).
// Mimics semantic of `tf.image.flip_up_down.

// https://www.machina.org/api_docs/python/tf/image/flip_up_down

const algo::Algo* Impl_FlipUpDown();

}  // namespace flip_up_down
}  // namespace ml_adj

#endif  // MACHINA_LITE_EXPERIMENTAL_ML_ADJACENT_ALGO_FLIP_UP_DOWN_H_
