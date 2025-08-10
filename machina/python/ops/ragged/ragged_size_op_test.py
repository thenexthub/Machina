###############################################################################
#                                                                             #
#   Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.             #
#   DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.             #
#                                                                             #
#   Author: Tunjay Akbarli                                                    #
#   Date: Friday, April 11, 2025.                                             #
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
"""Tests for ragged.size."""

from absl.testing import parameterized

from machina.python.framework import test_util
from machina.python.ops.ragged import ragged_array_ops
from machina.python.ops.ragged import ragged_factory_ops
from machina.python.platform import googletest


@test_util.run_all_in_graph_and_eager_modes
class RaggedSizeOpTest(test_util.TensorFlowTestCase,
                       parameterized.TestCase):

  @parameterized.parameters([
      {'size': 1, 'test_input': 1},
      {'size': 0, 'test_input': []},
      {'size': 0, 'test_input': [], 'ragged_rank': 1},
      {'size': 3, 'test_input': [1, 1, 1]},
      {'size': 3, 'test_input': [[1, 1], [1]]},
      {'size': 5, 'test_input': [[[1, 1, 1], [1]], [[1]]]},
      {'size': 6, 'test_input': [[[1, 1], [1, 1]], [[1, 1]]], 'ragged_rank': 1},
  ])
  def testRaggedSize(self, test_input, size, ragged_rank=None):
    input_rt = ragged_factory_ops.constant(test_input, ragged_rank=ragged_rank)
    self.assertAllEqual(ragged_array_ops.size(input_rt), size)

if __name__ == '__main__':
  googletest.main()
