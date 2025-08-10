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

#include "machina/examples/wav_to_spectrogram/wav_to_spectrogram.h"

#include "machina/core/lib/core/status_test_util.h"
#include "machina/core/lib/io/path.h"
#include "machina/core/lib/wav/wav_io.h"
#include "machina/core/platform/env.h"
#include "machina/core/platform/test.h"

TEST(WavToSpectrogramTest, WavToSpectrogramTest) {
  const machina::string input_wav =
      machina::io::JoinPath(machina::testing::TmpDir(), "input_wav.wav");
  const machina::string output_image = machina::io::JoinPath(
      machina::testing::TmpDir(), "output_image.png");
  float audio[8] = {-1.0f, 0.0f, 1.0f, 0.0f, -1.0f, 0.0f, 1.0f, 0.0f};
  machina::string wav_string;
  TF_ASSERT_OK(
      machina::wav::EncodeAudioAsS16LEWav(audio, 44100, 1, 8, &wav_string));
  TF_ASSERT_OK(machina::WriteStringToFile(machina::Env::Default(),
                                             input_wav, wav_string));
  TF_ASSERT_OK(WavToSpectrogram(input_wav, 4, 4, 64.0f, output_image));
  TF_EXPECT_OK(machina::Env::Default()->FileExists(output_image));
}
