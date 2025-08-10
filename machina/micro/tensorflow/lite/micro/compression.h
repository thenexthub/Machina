/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Friday, August 8, 2025.
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

#ifndef MACHINA_LITE_MICRO_MICRO_COMPRESSION_H_
#define MACHINA_LITE_MICRO_MICRO_COMPRESSION_H_

#ifdef USE_TFLM_COMPRESSION

#include "machina/lite/c/common.h"

namespace tflite {

//
// Compressed tensors
//

static constexpr const char* kCompressionMetadataString =
    "COMPRESSION_METADATA";

enum class CompressionScheme : uint8_t {
  kBinQuant,
};

struct LookupTableData {
  static constexpr size_t kMaxBitWidth = 7;
  static constexpr size_t kMaxValueTableChannelStride = 128;

  const void* value_table;             // Pointer into FlatBuffer Values.
  uint8_t value_table_channel_stride;  // elements per channel
  uint8_t compressed_bit_width : 3;    // 1 to 7 bits
  bool is_per_channel_quantized : 1;   // tensor is per-channel quantized
  bool use_alternate_axis : 1;         // shape default channel:
                                       // 0 = first, 1 = last
  uint8_t reserved : 3;
};

union CompressionData {
  LookupTableData* lut_data;
};

struct CompressionTensorData {
  CompressionScheme scheme;
  CompressionData data;
};

struct CompressedTensorList {
  // Sparsely populated array with the same number of elements as there are
  // tensors in the Subgraph. An alternative would include a tensor index in
  // the struct for each and walk the list on look up. This could be slow.
  const CompressionTensorData** tensors;
};

}  // namespace tflite

#endif  // USE_TFLM_COMPRESSION
#endif  // MACHINA_LITE_MICRO_MICRO_COMPRESSION_H_
