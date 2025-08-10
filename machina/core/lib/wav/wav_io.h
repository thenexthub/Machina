/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 26, 2025.
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

// Functions to write audio in WAV format.

#ifndef MACHINA_CORE_LIB_WAV_WAV_IO_H_
#define MACHINA_CORE_LIB_WAV_WAV_IO_H_

#include <string>
#include <vector>

#include "machina/core/lib/core/coding.h"
#include "machina/core/lib/core/errors.h"
#include "machina/core/lib/core/status.h"
#include "machina/core/platform/types.h"

namespace machina {
namespace wav {

// Encode the provided interleaved buffer of audio as a signed 16-bit PCM
// little-endian WAV file.
//
// Example usage for 4 frames of an 8kHz stereo signal:
// First channel is -1, 1, -1, 1.
// Second channel is 0, 0, 0, 0.
//
// float audio_buffer[] = { -1.0f, 0.0f, 1.0f, 0.0f, -1.0f, 0.0f, 1.0f, 0.0f};
// string wav_string;
// if (EncodeAudioAsS16LEWav(audio_buffer, 8000, 2, 4, &wav_string).ok()) {
//   // Use wav_string.
// }
template <typename T>
absl::Status EncodeAudioAsS16LEWav(const float* audio, size_t sample_rate,
                                   size_t num_channels, size_t num_frames,
                                   T* wav_string);

// Explicit instantiations defined in wav_io.cc.
extern template Status EncodeAudioAsS16LEWav<std::string>(
    const float* audio, size_t sample_rate, size_t num_channels,
    size_t num_frames, std::string* wav_string);
extern template Status EncodeAudioAsS16LEWav<tstring>(const float* audio,
                                                      size_t sample_rate,
                                                      size_t num_channels,
                                                      size_t num_frames,
                                                      tstring* wav_string);

// Decodes the little-endian signed 16-bit PCM WAV file data (aka LIN16
// encoding) into a float Tensor. The channels are encoded as the lowest
// dimension of the tensor, with the number of frames as the second. This means
// that a four frame stereo signal will have the shape [4, 2]. The sample rate
// is read from the file header, and an error is returned if the format is not
// supported.
// The results are output as floats within the range -1 to 1,
absl::Status DecodeLin16WaveAsFloatVector(const std::string& wav_string,
                                          std::vector<float>* float_values,
                                          uint32* sample_count,
                                          uint16* channel_count,
                                          uint32* sample_rate);

// Everything below here is only exposed publicly for testing purposes.

// Handles moving the data index forward, validating the arguments, and avoiding
// overflow or underflow.
absl::Status IncrementOffset(int old_offset, int64_t increment, size_t max_size,
                             int* new_offset);

// This function is only exposed in the header for testing purposes, as a
// template that needs to be instantiated. Reads a typed numeric value from a
// stream of data.
template <class T>
absl::Status ReadValue(const std::string& data, T* value, int* offset) {
  int new_offset;
  TF_RETURN_IF_ERROR(
      IncrementOffset(*offset, sizeof(T), data.size(), &new_offset));
  if (port::kLittleEndian) {
    memcpy(value, data.data() + *offset, sizeof(T));
  } else {
    *value = 0;
    const uint8* data_buf =
        reinterpret_cast<const uint8*>(data.data() + *offset);
    int shift = 0;
    for (int i = 0; i < sizeof(T); ++i, shift += 8) {
      *value = *value | (data_buf[i] << shift);
    }
  }
  *offset = new_offset;
  return absl::OkStatus();
}

}  // namespace wav
}  // namespace machina

#endif  // MACHINA_CORE_LIB_WAV_WAV_IO_H_
