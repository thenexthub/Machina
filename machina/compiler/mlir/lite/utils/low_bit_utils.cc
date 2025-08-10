/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, June 15, 2025.
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

#include "machina/compiler/mlir/lite/utils/low_bit_utils.h"

#include <cassert>
#include <cstdint>
#include <vector>

namespace tflite {

std::vector<uint8_t> PackInt4ValuesDensely(std::vector<uint8_t> src_buffer) {
  auto num_elements = src_buffer.size();
  auto packed_size = (num_elements + 1) / 2;
  std::vector<uint8_t> packed_buffer((num_elements + 1) / 2);

  for (int i = 0; i < num_elements - 1; i += 2) {
    packed_buffer[i / 2] = src_buffer[i] & 0x0F;
    packed_buffer[i / 2] |= src_buffer[i + 1] << 4;
  }

  // Copy the final nibble if the buffer is odd-lengthed
  if (num_elements % 2 != 0) {
    packed_buffer[packed_size - 1] = src_buffer[num_elements - 1] & 0x0F;
  }

  return packed_buffer;
}

std::vector<char> UnpackDenseInt4IntoInt8(
    const std::vector<uint8_t>& src_buffer, int64_t num_elements) {
  std::vector<char> unpacked_buffer;
  unpacked_buffer.reserve(num_elements);

  for (uint8_t value : src_buffer) {
    // Cast to signed before right-shifting to ensure correct sign extension
    unpacked_buffer.push_back(static_cast<int8_t>(value << 4) >> 4);
    unpacked_buffer.push_back(static_cast<int8_t>(value) >> 4);
  }

  // The last element might be a padded zero, so check and pop if needed
  if (unpacked_buffer.size() > num_elements) {
    assert(unpacked_buffer.size() == num_elements + 1);
    unpacked_buffer.pop_back();
  }

  return unpacked_buffer;
}

}  // namespace tflite
