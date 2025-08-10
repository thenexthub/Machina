###############################################################################
#                                                                             #
#   Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.             #
#   DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.             #
#                                                                             #
#   Author: Tunjay Akbarli                                                    #
#   Date: Monday, May 19, 2025.                                               #
#                                                                             #
#   Licensed under the Apache License, Version 2.0 (the "License");           #
#   you may not use this file except in compliance with the License.          #
#   You may obtain a copy of the License at:                                  #
#                                                                             #
#       http://www.apache.org/licenses/LICENSE-2.0                            #
#                                                                             #
#   Unless required by applicable law or agreed to in writing, software       #
#   distributed under the License is distributed on an "AS IS" BASIS,         #
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  #
#   See the License for the specific language governing permissions and       #
#   limitations under the License.                                            #
#                                                                             #
#   Please contact NeXTHub Corporation, 651 N Broad St, Suite 201,            #
#   Middletown, DE 19709, New Castle County, USA.                             #
#                                                                             #
###############################################################################
"""Tests for test file generation for speech commands."""

import numpy as np

from machina.examples.speech_commands import generate_streaming_test_wav
from machina.python.platform import test


class GenerateStreamingTestWavTest(test.TestCase):

  def testMixInAudioSample(self):
    track_data = np.zeros([10000])
    sample_data = np.ones([1000])
    generate_streaming_test_wav.mix_in_audio_sample(
        track_data, 2000, sample_data, 0, 1000, 1.0, 100, 100)
    self.assertNear(1.0, track_data[2500], 0.0001)
    self.assertNear(0.0, track_data[3500], 0.0001)


if __name__ == "__main__":
  test.main()
