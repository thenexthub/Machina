###############################################################################
#                                                                             #
#   Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.             #
#   DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.             #
#                                                                             #
#   Author: Tunjay Akbarli                                                    #
#   Date: Saturday, May 31, 2025.                                             #
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
"""Tests for ragged.rank op."""

from absl.testing import parameterized
from machina.python.framework import test_util
from machina.python.ops.ragged import ragged_array_ops
from machina.python.ops.ragged import ragged_factory_ops
from machina.python.platform import googletest


@test_util.run_all_in_graph_and_eager_modes
class RaggedRankOpTest(test_util.TensorFlowTestCase,
                       parameterized.TestCase):

  @parameterized.parameters([
      # Rank 0
      dict(
          test_input=1,
          expected_rank=0,
      ),
      # Rank 1
      dict(
          test_input=[1],
          expected_rank=1,
      ),
      dict(
          test_input=[1, 2, 3, 4],
          expected_rank=1,
      ),
      # Rank 2
      dict(
          test_input=[[1], [2], [3]],
          expected_rank=2,
      ),
      # Rank 3
      dict(
          test_input=[[[1], [2, 3]], [[4], [5, 6, 7]]],
          expected_rank=3,
      ),
      # Rank 3, ragged_rank=2
      dict(
          test_input=[[[1], [2, 3], [10, 20]],
                      [[4], [5, 6, 7]]],
          expected_rank=3,
          ragged_rank=2,
      ),
      # Rank 4, ragged_rank=3 with dimensions: {2, (1, 2), (2), (1, 2)}
      dict(
          test_input=[[[[1], [2]]],
                      [[[3, 4], [5, 6]], [[7, 8], [9, 10]]]],
          expected_rank=4,
      ),
      # Rank 4, ragged_rank=2 with dimensions: {2, (1, 2), (1, 2), 2}
      dict(
          test_input=[
              [[[1, 2]]],
              [[[5, 6], [7, 8]],
               [[9, 10], [11, 12]]]],
          expected_rank=4,
          ragged_rank=2,
      ),

  ])
  def testRaggedRank(self, test_input, expected_rank, ragged_rank=None):
    test_input = ragged_factory_ops.constant(
        test_input, ragged_rank=ragged_rank)
    self.assertAllEqual(ragged_array_ops.rank(
        test_input), expected_rank)


if __name__ == '__main__':
  googletest.main()
