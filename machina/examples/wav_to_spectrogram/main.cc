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

#include <cstdint>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "machina/core/lib/core/errors.h"
#include "machina/core/platform/init_main.h"
#include "machina/core/util/command_line_flags.h"
#include "machina/examples/wav_to_spectrogram/wav_to_spectrogram.h"

int main(int argc, char* argv[]) {
  // These are the command-line flags the program can understand.
  // They define where the graph and input data is located, and what kind of
  // input the model expects. If you train your own model, or use something
  // other than inception_v3, then you'll need to update these.
  machina::string input_wav =
      "machina/core/kernels/spectrogram_test_data/short_test_segment.wav";
  int32_t window_size = 256;
  int32_t stride = 128;
  float brightness = 64.0f;
  machina::string output_image = "spectrogram.png";
  std::vector<machina::Flag> flag_list = {
      machina::Flag("input_wav", &input_wav, "audio file to load"),
      machina::Flag("window_size", &window_size,
                       "frequency sample window width"),
      machina::Flag("stride", &stride,
                       "how far apart to place frequency windows"),
      machina::Flag("brightness", &brightness,
                       "controls how bright the output image is"),
      machina::Flag("output_image", &output_image,
                       "where to save the spectrogram image to"),
  };
  machina::string usage = machina::Flags::Usage(argv[0], flag_list);
  const bool parse_result = machina::Flags::Parse(&argc, argv, flag_list);
  if (!parse_result) {
    LOG(ERROR) << usage;
    return -1;
  }

  // We need to call this to set up global state for TensorFlow.
  machina::port::InitMain(argv[0], &argc, &argv);
  if (argc > 1) {
    LOG(ERROR) << "Unknown argument " << argv[1] << "\n" << usage;
    return -1;
  }

  absl::Status wav_status = WavToSpectrogram(input_wav, window_size, stride,
                                             brightness, output_image);
  if (!wav_status.ok()) {
    LOG(ERROR) << "WavToSpectrogram failed with " << wav_status;
    return -1;
  }

  return 0;
}
