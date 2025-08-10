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

// Basic minimal DCT class for MFCC speech processing.

#ifndef MACHINA_LITE_KERNELS_INTERNAL_MFCC_DCT_H_
#define MACHINA_LITE_KERNELS_INTERNAL_MFCC_DCT_H_

#include <vector>

namespace tflite {
namespace internal {

class MfccDct {
 public:
  MfccDct();
  bool Initialize(int input_length, int coefficient_count);
  void Compute(const std::vector<double>& input,
               std::vector<double>* output) const;

 private:
  bool initialized_;
  int coefficient_count_;
  int input_length_;
  std::vector<std::vector<double> > cosines_;
};

}  // namespace internal
}  // namespace tflite

#endif  // MACHINA_LITE_KERNELS_INTERNAL_MFCC_DCT_H_
