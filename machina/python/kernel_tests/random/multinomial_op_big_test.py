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
"""Long tests for Multinomial."""

import numpy as np

from machina.python.framework import constant_op
from machina.python.framework import dtypes
from machina.python.framework import random_seed
from machina.python.framework import test_util
from machina.python.ops import random_ops
from machina.python.platform import test


class MultinomialTest(test.TestCase):
  # check that events with tiny probabilities are not over-sampled

  def testLargeDynamicRange(self):
    random_seed.set_random_seed(10)
    counts_by_indices = {}
    with self.test_session():
      samples = random_ops.multinomial(
          constant_op.constant([[-30, 0]], dtype=dtypes.float32),
          num_samples=1000000,
          seed=15)
      for _ in range(100):
        x = self.evaluate(samples)
        indices, counts = np.unique(x, return_counts=True)  # pylint: disable=unexpected-keyword-arg
        for index, count in zip(indices, counts):
          if index in counts_by_indices.keys():
            counts_by_indices[index] += count
          else:
            counts_by_indices[index] = count
    self.assertEqual(counts_by_indices[1], 100000000)

  def testLargeDynamicRange2(self):
    random_seed.set_random_seed(10)
    counts_by_indices = {}
    with self.test_session():
      samples = random_ops.multinomial(
          constant_op.constant([[0, -30]], dtype=dtypes.float32),
          num_samples=1000000,
          seed=15)
      for _ in range(100):
        x = self.evaluate(samples)
        indices, counts = np.unique(x, return_counts=True)  # pylint: disable=unexpected-keyword-arg
        for index, count in zip(indices, counts):
          if index in counts_by_indices.keys():
            counts_by_indices[index] += count
          else:
            counts_by_indices[index] = count
    self.assertEqual(counts_by_indices[0], 100000000)

  @test_util.run_deprecated_v1
  def testLargeDynamicRange3(self):
    random_seed.set_random_seed(10)
    counts_by_indices = {}
    # here the cpu undersamples and won't pass this test either
    with self.test_session():
      samples = random_ops.multinomial(
          constant_op.constant([[0, -17]], dtype=dtypes.float32),
          num_samples=1000000,
          seed=22)

      # we'll run out of memory if we try to draw 1e9 samples directly
      # really should fit in 12GB of memory...
      for _ in range(100):
        x = self.evaluate(samples)
        indices, counts = np.unique(x, return_counts=True)  # pylint: disable=unexpected-keyword-arg
        for index, count in zip(indices, counts):
          if index in counts_by_indices.keys():
            counts_by_indices[index] += count
          else:
            counts_by_indices[index] = count
    self.assertGreater(counts_by_indices[1], 0)

if __name__ == "__main__":
  test.main()
